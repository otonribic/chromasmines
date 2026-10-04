import struct

def encode_ase_string(text: str) -> bytes:
    """Encodes a string as a null-terminated UTF-16-BE string prefixed by length (in characters)."""
    # Number of characters, including the trailing null terminator
    char_len = len(text) + 1
    # UTF-16-BE encoding
    encoded_str = text.encode("utf-16-be") + b"\x00\x00"
    return struct.pack(">H", char_len) + encoded_str

def create_ase_color_block(name: str, r: int, g: int, b: int) -> bytes:
    """
    Encodes an RGB color block.
    ASE stores color channels as 32-bit big-endian floats scaled from 0.0 to 1.0.
    """
    # Encode block type (0x0001 for Color)
    block_type = b"\x00\x01"
    
    # Color payload construction
    encoded_name = encode_ase_string(name)
    color_model = b"RGB "  # 4-byte ASCII identifier for RGB
    
    # Convert 0-255 RGB integers to 0.0-1.0 32-bit floats
    r_float = r / 255.0
    g_float = g / 255.0
    b_float = b / 255.0
    
    # Color type: 0 = Global, 1 = Spot, 2 = Process
    color_type = struct.pack(">h", 0)
    
    payload = (
        encoded_name + 
        color_model + 
        struct.pack(">fff", r_float, g_float, b_float) + 
        color_type
    )
    
    # 4-byte length header for the block payload
    block_length = struct.pack(">I", len(payload))
    
    return block_type + block_length + payload

def create_ase_group_blocks(group_name: str, colors: list[tuple[str, int, int, int]]) -> bytes:
    """Encodes a palette group containing start block, color blocks, and end block."""
    blocks = bytearray()
    
    # Group Start Block (0xc001)
    group_name_bytes = encode_ase_string(group_name)
    group_start_len = struct.pack(">I", len(group_name_bytes))
    blocks.extend(b"\xc0\x01" + group_start_len + group_name_bytes)
    
    # Add individual color blocks
    for name, r, g, b in colors:
        blocks.extend(create_ase_color_block(name, r, g, b))
        
    # Group End Block (0xc002, payload length 0)
    blocks.extend(b"\xc0\x02\x00\x00\x00\x00")
    
    return bytes(blocks)

def write_ase_file(filename: str, palette_name: str, colors: list[tuple[str, int, int, int]]) -> None:
    """Writes header, group metadata, and color blocks out to an .ase file."""
    # ASE Header: 'ASEF' magic bytes + Major Version 1 + Minor Version 0
    header = b"ASEF" + struct.pack(">HH", 1, 0)
    
    # Calculate total block count: 1 group start + N colors + 1 group end
    total_blocks = len(colors) + 2
    block_count_bytes = struct.pack(">I", total_blocks)
    
    # Encode body
    body = create_ase_group_blocks(palette_name, colors)
    
    with open(filename, "wb") as f:
        f.write(header + block_count_bytes + body)

# -------------------------------------------------------------------
# Example Usage: Generating a 6-color palette
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_colors = [
        ("Midnight Blue", 25, 42, 86),
        ("Emerald Green", 46, 204, 113),
        ("Sunflower Yellow", 241, 196, 15),
        ("Coral Red", 231, 76, 60),
        ("Amethyst Purple", 155, 89, 182),
        ("Cloud White", 236, 240, 241)
    ]

    output_filename = "custom_palette.ase"
    write_ase_file(output_filename, "My 6-Color Set", sample_colors)
    print(f"Successfully generated '{output_filename}' with {len(sample_colors)} colors.")