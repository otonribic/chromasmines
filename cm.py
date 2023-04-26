'''
CHROMAS MINES

Python 3.7

Dark Forces Palette & Colormap editor
by
Fish (oton.ribic@bug.hr), 2021-2023
Cindy Winter (xxxxx.xxxxxx@xxxxxx.xx.xx), 2021

Python 3.7

To Add for 0.9:
X progress bar on map autocalculation
X load only part of palette when merging, maybe also mix instead of replace, possibly with offset
X merge color map (load only a segment instead of all)
X update Nar Shaddaa colormap from Textures.gob
X write hints for map autocalc

For 0.94:
X add "disable flashlight" in colormap save
X auto-preview paletted BMP's when changing brightness
X allow mix of ranges and individual colors for "solve unevennesses"
X "Save as" on "save" even if filenames specified
X allow reverse search of color candidates in auto map calc
X Mass replace: enter colors to be replaced with other colors in the colormap

For 0.95
X import image as an approximated colormap
X export colormap as an image
X resize window ([,],Menu)
X progressive saturation

For 0.96
X reverse sort

For 0.97
- finalize help video link
X flashlight editor and slider

For 0.98
X color themes

For 0.99
X add label about engine color behavior in colormap

For 0.100
X color map auto homogenization

1.0 RELEASE

1.1 RELEASE
X raindrop auto-generate method
X use arrow keys for vertical navigation
X can leave with Q now
X Phrik Freak is the new default color theme
X Added unbalance
X Added 'make the current color luminous'
X Added Export to paletted image
X Added DF-21 link
X Tearable menus
X Psychodelic hue shift algorithm for autocalculation
'''

import webbrowser
import PySimpleGUI as sg
import cmconst
import copy
from PIL import Image as pil
import colorsys

# GENERAL PSEUDO-CONSTANTS

DEBUG_MODE = True

APPNAME = 'Chromas Mines'
APPVERSION = '1.1'
ICONFILE = 'cm.ico'
COLORTHEME = 'Phrik Freak (Default)'  # Default color theme
FONT_PALETTE = 'Tahoma'
FONT_PALETTE_SIZE = 8
SWATCH_BORDER = 0  # Width of lines
CURSOR_COLOR = '#FFFF00'  # On color map
FONT_WINDOW = 'Tahoma'
FONT_WINDOW_SIZE = 10
UNDO_LEVELS = 20
POPUP_BACKCOLOR = '#040313'
REVERSE_LIGHT = False  # Whether colormap light is oriented downwards or upwards
COLOR_MATCHING_PRIO_FACTOR = 5  # Factor by which the priority picking will be pondered
SCALE = 32  # Multiplier which will determine the pixel size of window(s)

# AUXILIARIES \1

# Placeholder global variables
Main = None


def _get_unused_themes():
    '''[INTERNAL!] Show themes currently unused in CM.'''
    agg = []
    for k in cmconst.COLORTHEMES.keys():
        agg.append(cmconst.COLORTHEMES[k])
    for theme in sg.theme_list():
        if theme not in agg:
            print(theme, end=', ')
    quit()


def popup(text):
    '''Show a generic popup with desired (possibly multiline) text.'''
    sg.popup_no_titlebar(text)


def norm255(value):
    '''Normalize given value to integer 0-255.'''
    value = round(float(value))
    if value < 0: value = 0
    if value > 255: value = 255
    return value


def gethexcol(r, g, b):
    '''Get convenient #RRGGBB hex code for a RGB 0-255 color.'''
    # First make sure the inputs make sense
    r = norm255(r)
    g = norm255(g)
    b = norm255(b)
    # Now format and return
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)


def get_contrast_color(r, g, b):
    '''Get a color for drawing text on a given background color.
    Return it in the convenient #001122 format.'''
    if r + g + b > 384: ccol = '#000000'
    else: ccol = '#FFFFFF'
    return ccol


def load_palette(file):
    '''Load a palette from a .PAL file and return a palette-formatted list of lists.'''
    inp = open(file, 'rb')
    fraw = inp.read()[0:768]  # Sanity measure
    inp.close()
    # Combine 768 bytes to 256 RGB triplets
    parsed = [[fraw[d] * 4, fraw[d + 1] * 4, fraw[d + 2] * 4] for d in range(0, 767, 3)]
    return parsed


def save_palette(palette, filename):
    '''Save a given palette (list of RGB lists) to the DF palette of given path.'''
    outf = open(filename, 'wb')
    palcontent = []  # Collector
    for rgb in palette:
        # Divide by 4 (DF uses 6-bit palettes)
        rgb = [e // 4 for e in rgb]
        palcontent.extend(rgb)
    outf.write(bytes(palcontent))
    outf.close()


def load_colormap(file):
    '''Load a colormap from a .CMP file and return colormap-formatted list of lists.'''
    inp = open(file, 'rb')
    fraw = inp.read()[0:8192]  # Sanity measure
    inp.close()
    # Combine 8192 bytes to 256 32-plets
    parsed = [list(fraw[d::256]) for d in range(0, 256)]
    return parsed


def save_colormap(colormap, filename, flashlight=50):
    '''Save a given colormap (list of light maps) to the DF colormap of given path.
    If flashlight given, it specifies its intensity in percentage, with 50 being
    the perfect linear average (a la NARSHADA.CMP).
    '''
    outf = open(filename, 'wb')
    cmpcontent = []  # Collector
    # Iterate to fill over lights, then inner, to color index
    for lightlevel in range(32):
        for color in range(256):
            cmpcontent.append(colormap[color][lightlevel])
    # Add a remainder 128 bytes ranging from 0 to 31 depending on flashlight function
    cmpcontent.extend(get_flashlight_range(flashlight))
    # Convert and save
    outf.write(bytes(cmpcontent))
    outf.close()


def draw_palette(palette):
    '''Redraw palette in the main window palette view and add index numbers.'''
    Main['palette'].erase()
    counter = 0
    for r, g, b in palette[0:256]:
        # Determine swatch position
        sx = (counter % 32) * 32
        sy = counter // 32 * 32
        # Draw with some inner padding (1 px all sides)
        Main['palette'].draw_rectangle((sx, sy), (sx + 32, sy + 32),
                                       '#{0:02x}{1:02x}{2:02x}'.format(r, g, b),
                                       line_width=SWATCH_BORDER)
        # Calculate contrasting color for index number
        fontcol = get_contrast_color(r, g, b)
        # And the index number
        Main['palette'].draw_text(str(counter), location=(sx + 16, sy + 16), color=fontcol,
                                  text_location='center', font=[FONT_PALETTE, FONT_PALETTE_SIZE])
        counter += 1
    # Draw selected color's border
    sx = (Mselcolor % 32) * 32
    sy = Mselcolor // 32 * 32
    curcolor = get_contrast_color(*Mpalette[Mselcolor])
    Main['palette'].draw_rectangle((sx + 1, sy + 1), (sx + 31, sy + 31),
                                   line_color=curcolor, line_width=2)


def draw_colormap(colormap):
    '''Redraw colormap in the main window colormap view.'''
    Main['colormap'].erase()
    counter = 0
    for seq in colormap:
        # Get position in the map
        xpos = counter * 4
        for id, link in enumerate(seq):
            ypos = id * 8
            # Get local color
            lcol = '#{0:02x}{1:02x}{2:02x}'.format(*Mpalette[link])
            Main['colormap'].draw_rectangle((xpos, ypos), (xpos + 4, ypos + 8),
                                            lcol, line_width=SWATCH_BORDER)
            # Is it a selected color?
            if link == Mselcolor and Mshowselinmap:
                Main['colormap'].draw_rectangle((xpos, ypos), (xpos + 3, ypos + 7),
                                                lcol, line_width=1,
                                                line_color=CURSOR_COLOR)
        counter += 1
    # Draw selected color's map
    xpos = Mselcolor * 4
    Main['colormap'].draw_rectangle((xpos, 0), (xpos + 4, 255), line_color=CURSOR_COLOR)


def draw_light_range(lrange):
    Main['lightrange'].erase()
    # Iterate over 0-31 range
    for id, clr in enumerate(lrange[0:32]):
        # Actual color
        Main['lightrange'].draw_rectangle((id * 32, 0), (id * 32 + 32, 16),
                                          fill_color='#{0:02x}{1:02x}{2:02x}'.
                                          format(*Mpalette[clr]),
                                          line_width=SWATCH_BORDER)
        # ID's of selected colors
        Main['lightrange'].draw_text(str(clr), location=(id * 32 + 16, 7), text_location='center',
                                     color=get_contrast_color(*Mpalette[clr]),
                                     font=[FONT_PALETTE, FONT_PALETTE_SIZE])
        # Corresponding lightness
        Main['lightrange'].draw_rectangle((id * 32, 16), (id * 32 + 32, 32),
                                          fill_color='#{0:02x}{1:02x}{2:02x}'.
                                          format(id * 8, id * 8, id * 8),
                                          line_width=SWATCH_BORDER)
        # Mark of light level
        if id <= 15: fcol = '#FFFFFF'
        else: fcol = '#000000'
        Main['lightrange'].draw_text(str(id), location=(id * 32 + 16, 23), text_location='center',
                                     color=fcol, font=[FONT_PALETTE, FONT_PALETTE_SIZE])


def update_title():
    '''Update the title of the main screen to show filenames.'''
    Main.set_title(APPNAME + ' ' + APPVERSION + ' - ' + Mfiles[0] + ' • ' + Mfiles[1])


def update_selcolor_def():
    '''Update the selected color displayed definition.'''
    Main['SCol'].update('Selected color - Index: {0}    RGB: {1}, {2}, {3}'.
                        format(Mselcolor, *Mpalette[Mselcolor])
                        )


def draw_all():
    '''Redraw the entire app window.'''
    draw_palette(Mpalette)
    draw_colormap(Mcolormap)
    draw_light_range(Mcolormap[Mselcolor])
    update_title()
    update_selcolor_def()


def init_Main_window():  # \Main
    global Main
    '''Initialize PySimpleGUI main window.'''

    sg.theme(cmconst.COLORTHEMES[COLORTHEME])

    # Menu
    MENUlevels = [entry[0] + '::' + entry[1] for entry in cmconst.ORIGS]
    MENUlayout = [['&File', ['&New (N)', '&Open... (O)', '&Save (S)', 'Save &as... (A)',
                             '---', '&Quit']],
                  ['&Edit', ['&Undo (U)', 'Toggle swatch &borders (B)',
                             '&Reverse light orientation (-)', '---',
                             'Window &size... (Z)', 'Window &theme...'], ],
                  ['&Palette', ['&Tint... (T)', 'General &adjustments... (J)', 'Auto-&gradient... (G)',
                                'Progressive &saturation...',
                                '---',
                                '&Import from paletted image...', 'E&xport to paletted image...',
                                '&Merge with... (M)', '&Sort...',
                                '---',
                                '&Copy selected color... (C)',
                                '&Redundancy and consistency checks... (F9)',
                                'Coverage scatter &plot...']],
                  ['&Map', ['&Auto-calculate... (F2)', '&Solve unevennesses...',
                            'Linear &homogenization...', '&Unbalance...', '---',
                            '&Preview bitmap... (F3)', '&Replace map with... (F4)',
                            '&Mass replace colors... (F8)', 'E&xport to image...',
                            '&Load from image...', '---', 'Ma&ke selected color luminous (L)',
                            '&Toggle selected color (\\)', ]],
                  ['Pre&sets', MENUlevels],
                  ['&Help',
                   ['&Tutorial...', '&About...', '&Dark Forces community...', '&Whys and becauses...',
                    '&Keyboard shortcuts...']],
                  ]
    Wlayout = [[sg.Menu(MENUlayout, tearoff=True, key='menu')],
               [sg.T('Current palette')],
               [sg.Graph(canvas_size=(32 * SCALE, 8 * SCALE), background_color='#111111', key='palette',
                         graph_bottom_left=(0, 256), graph_top_right=(1024, 0), enable_events=True)],
               [sg.T('Selected color - Index: 0    RGB: 0, 0, 0             ', key='SCol')],
               [sg.Button('◄ Prev', key='PreviousColor'), sg.Button('Next ►', key='NextColor'),
                sg.Button('Modify color'), ],
               [sg.T('Color light map')],
               [sg.Graph(canvas_size=(32 * SCALE, 8 * SCALE), background_color='#000000', key='colormap',
                         graph_bottom_left=(0, 256), graph_top_right=(1024, 0), enable_events=True)],
               [sg.T("\nSelected color's light mapping")],
               [sg.Graph(canvas_size=(32 * SCALE, SCALE), background_color='#000000', key='lightrange',
                         graph_bottom_left=(0, 32), graph_top_right=(1024, 0), enable_events=True)],
               ]
    Main = sg.Window(
        APPNAME,
        layout=Wlayout,
        icon=ICONFILE,
        resizable=False,
        finalize=True,
        element_justification='center',
        font=(FONT_WINDOW, FONT_WINDOW_SIZE),
        margins=(SCALE / 2, SCALE / 2),
        return_keyboard_events=True,)
    draw_all()


def save_undo():
    '''Add the current palette and colormap to the undo register.'''
    global Mundo
    # Deep copy to avoid reference conundrums
    Mundo.append((copy.deepcopy(Mpalette), copy.deepcopy(Mcolormap)))
    # Reached the maximum?
    while len(Mundo) > UNDO_LEVELS: Mundo = Mundo[1:]


def undo():
    '''Revert to the latest living palette and colormap from the Mundo register.'''
    global Mpalette
    global Mcolormap
    global Mundo
    # Is there anything to undo in the first place?
    if not Mundo:
        popup('Bugger! No further undoing is possible.')
        return
    # There IS something - let's apply the Undoing
    Mpalette = Mundo[-1][0]
    Mcolormap = Mundo[-1][1]
    # Snip that one away
    Mundo = Mundo[:-1]
    draw_all()


def calc_tint(operation=0, targetrgb=(0, 0, 0), mix=50):
    '''Calculate tinted palette of Mpalette, with given:
    operation: 0=mix, 1=add, 2=subtract
    targetrgb: the RGB values if using mix
    mix: the midpoint of mixing (0-100) if using mix
    '''
    workp = copy.deepcopy(Mpalette)  # Copy palette

    for id, rgb in enumerate(workp):  # Process each
        # Iterate over channels
        if operation == 0:
            # Mix
            rgb = [norm255(rgb[c] - (rgb[c] - targetrgb[c]) * mix / 100) for c in (0, 1, 2)]
        elif operation == 1:
            # Add
            rgb = [norm255(rgb[c] + targetrgb[c]) for c in (0, 1, 2)]
        elif operation == 2:
            # Subtract
            rgb = [norm255(rgb[c] - targetrgb[c]) for c in (0, 1, 2)]
        workp[id] = rgb  # Update

    return workp


def palette_check():
    '''Performs various checks of sensibility and quality of the palettes and returns them as
    list of lists, each element for its own problem (if encountered, otherwise []).
    0: list of pairs of duplicate colors (taking into account final 6-bit conversion!)
    1: list of colors unused in the colormap
    '''

    # Step 1: Find duplicates
    duplicates = []  # Local collector
    pal = copy.deepcopy(Mpalette)  # Create a working instance
    pal = [(r // 4, g // 4, b // 4) for r, g, b in pal]  # Create DF 6-bit colors
    # Triangular search
    for k1 in range(256):
        # Skip special colors
        if k1 in range(24, 32): continue
        for k2 in range(k1 + 1, 256):
            if pal[k1] == pal[k2]: duplicates.append((k1, k2))

    # Step 2: Find unused colors in mapping
    used = set()  # Collector of used colors
    # Find all used
    for color in range(256):
        for lightness in range(32):
            used.add(Mcolormap[color][lightness])
    # Find 0..255 inversion
    unused = [e for e in range(256) if e not in used]

    # Collect all together and return it
    return [duplicates, unused]


def calc_adjustments(brig, cont, gamma, sat):
    '''Get Mpalette and apply various standard adjustments, based on:
    brightness -255..255
    contrast -50..50
    gamma -50..50
    saturation -50..50
    and return it.'''

    wpal = copy.deepcopy(Mpalette)  # Work palette

    # Iterate over all
    for col in range(0, 256):
        ec = wpal[col]  # Edited color
        # Apply brightness
        if brig:
            ec = [norm255(e + brig) for e in ec]
        # Apply contrast
        if cont:
            ec = [norm255((e - 128) * ((cont + 50) / 50) + 128) for e in ec]
        # Apply gamma
        if gamma:
            ec = [norm255(((e / 255 + 0.001)**(1 - gamma / 50)) * 255) for e in ec]
        # Apply saturation
        if sat:
            avrg = sum(ec) / 3
            ec = [norm255((e - avrg) * ((sat + 50) / 50) + avrg) for e in ec]
        # Return the color to the palette
        wpal[col] = ec

    return wpal


def generate_gradient(fcol, lcol):
    '''Generate a gradient on Mpalette starting with fcol and ending with lcol (includive).'''
    # Check if the order is reverse - correct it if so
    if fcol > lcol: fcol, lcol = lcol, fcol
    # Create work palette
    wpal = copy.deepcopy(Mpalette)
    # Get actual values
    frgb = Mpalette[fcol]
    lrgb = Mpalette[lcol]
    # Calculate deltas
    delta = [lrgb[c] - frgb[c] for c in (0, 1, 2)]
    # Number of steps
    span = lcol - fcol
    steps = [delta[c] / span for c in (0, 1, 2)]
    # Iterate over the span
    for step in range(1, span):  # First and last not needed as they stay the same
        # Get the color value
        new = [norm255(frgb[c] + steps[c] * step) for c in (0, 1, 2)]
        # Replace it in work palette
        wpal[fcol + step] = new
    return wpal


def calc_color_map(
        applycolorrange,
        applylightrange,
        projection,
        method,
        priority,
        keepluminescent=True,
        ignore2431=True,
        reverseorder=False):
    '''
    Calculate the color map (the main job of this program!) and return it (do not change
    anything in place). Parameters:
    applycolorrange - range of colors in the map to consider in the first place
    applylightrange - range of light levels in the map to consider in the first place
    projection - (min, max) range of PROJECTED lights in the given area
    method - 1: straight from full darkness (black) to the desired color
             2: from the currently selected color to the desired color
             3: from full darkness to the selected color used as a mask (water effect)
             4: raindrop trickle algorithm (ignores light ranges)
             5: psychodelic hue shift (ignores light ranges)
    priority - 0: find the nearest color to the calculated one via RGB
               1: find the nearest color to the calculated one via HSB
               2: prioritize correct hue
               3: deprioritize correct saturation
    keepluminescent - assume light is always 31 for colors 0-23
    ignore2431 - never modify colors in range 24-31
    reverseorder - search for suitable colors 'backwards'
    '''
    # First create a work map
    wmap = copy.deepcopy(Mcolormap)
    # Get deltas for projected light calculations
    projdelta = projection[1] - projection[0] + 1
    # Calculate projection step for each map light range
    projstep = projdelta / len(applylightrange)

    # Iterate over all needed colors
    for counter, mapcol in enumerate(applycolorrange):
        if DEBUG_MODE: print('Automap:', mapcol)
        # Prog bar
        if not sg.one_line_progress_meter('Calculating map...', current_value=counter + 1,
                                          max_value=len(applycolorrange), key='autocalcprog',
                                          orientation='h',):
            if counter < len(applycolorrange) - 1:  # Because it returns False if at end!
                return Mcolormap  # Return what is "living" as a map anyway if cancelled
        # Some pre-calculation necessities
        # RGB the range will start from
        if method == 2:
            sourcergb = Mpalette[Mselcolor]
        else:
            sourcergb = [0, 0, 0]
        # Find the target color
        if method == 3:
            workcol = Mpalette[mapcol]
            workcol = [max(workcol), max(workcol), max(workcol)]
            workcol = [workcol[c] * Mpalette[Mselcolor][c] / 255 for c in (0, 1, 2)]
            targetrgb = [norm255(c) for c in workcol]
        else:
            targetrgb = Mpalette[mapcol]  # This RGB will be used as a (possible) target
        # Find the deltas
        deltargb = [targetrgb[c] - sourcergb[c] for c in (0, 1, 2)]

        # Now iterate over lights with given start and end colors
        if method in [1, 2, 3]:  # Standard linear approximators
            for step, light in enumerate(applylightrange):
                # Deal with some special cases
                # Luminescents, just keep original color
                if keepluminescent and mapcol <= 23:
                    wmap[mapcol][light] = mapcol
                    continue
                # Ignored colors
                if ignore2431 and mapcol >= 24 and mapcol <= 31:
                    continue

                # Special cases taken care of, resume to do actual color picking from here
                loclight = round(projection[0] + projstep * step)
                mixlight = loclight / 31  # Get 0..1 for easier "mixing"
                # Mixlight is now the transition 0..1 from source- to targetrgb
                # Get the color
                mixed = [norm255(sourcergb[c] + deltargb[c] * mixlight) for c in (0, 1, 2)]
                # Now find the best matching color in the palette to 'mixed'
                nearest = get_nearest_color(mixed, priority, reverseorder)
                # Finally, update in the colormap
                wmap[mapcol][light] = nearest

        # Special raintrickle method does not iterate over lights directly
        if method == 4:
            # This iterates from the lightest to the darkest
            # but descend from the brightest (real) color

            wmap[mapcol][31] = mapcol
            for light in range(30, -1, -1):

                # Deal with some special cases
                # Luminescents, just keep original color
                if keepluminescent and mapcol <= 23:
                    wmap[mapcol][light] = mapcol
                    continue
                # Ignored colors
                if ignore2431 and mapcol >= 24 and mapcol <= 31:
                    continue

                # Get the reference color
                reference = Mpalette[wmap[mapcol][light + 1]]
                rr, rg, rb = reference
                mono = rr == rg == rb  # Special case if it's monochromatic
                # Find nearest darker color (yet not the same one!)
                bestcol = 0  # Index of the best color
                nearest = 768  # Best match so far
                for search in range(0, 256):
                    ser, seg, seb = Mpalette[search]
                    lightdiff = sum([ser - rr, seg - rg, seb - rb])
                    if lightdiff >= 0:
                        # The checked color is same or brighter; discard it
                        continue
                    # Calculate total difference
                    lightdiff = sum([abs(ser - rr), abs(seg - rg), abs(seb - rb)])
                    # Check the difference
                    if lightdiff < nearest:
                        # Check if the special case of monochromeness
                        if mono and not (ser == seg == seb): continue
                        # Found a better match
                        nearest = lightdiff
                        bestcol = search
                # Fallback to the same reference color if no better candidates were found
                if nearest == 768:
                    bestcol = wmap[mapcol][light + 1]
                wmap[mapcol][light] = bestcol

        # Hue shift over the light range
        if method == 5:
            for light in range(32):
                # Deal with some special cases
                # Luminescents, just keep original color
                if keepluminescent and mapcol <= 23:
                    wmap[mapcol][light] = mapcol
                    continue
                # Ignored colors
                if ignore2431 and mapcol >= 24 and mapcol <= 31:
                    continue

                # Get reference RGB
                rr, rg, rb = Mpalette[mapcol]
                # Calculate HSV offset
                h, s, v = colorsys.rgb_to_hsv(rr, rg, rb)
                h += light / 32
                tr, tg, rb = colorsys.hsv_to_rgb(h, s, v)
                # Find match
                nearest = get_nearest_color((tr, tg, rb), 0)
                # Update in the temporary colormap
                wmap[mapcol][light] = nearest

    return wmap


def get_nearest_color(rgb, priority=0, reverse=False):
    '''
    Scan through the Mpalette and find the closest matching color to the RGB values supplied.
    Priority: 0 - find the nearest color via RGB
              1 - find the nearest color via HSB
              2 - give priority to correct hue
              3 - give priority to correct saturation
    If reverse is specified, look for colors in the reverse order (prefer 'higher').
    Return the color's index number from Mpalette.
    '''
    # Choose range to be searched
    if reverse:
        crange = range(255, -1, -1)
    else:
        crange = range(0, 256, 1)
    # Now further depends on the type of search
    if priority == 0:
        # Direct straightforward RGB
        # Set comparator
        bestdelta = 769
        bestcolor = 0
        # Scan through the colors
        for id in crange:
            pcolor = Mpalette[id]
            delta = sum([abs(rgb[c] - pcolor[c]) for c in (0, 1, 2)])
            if delta < bestdelta:
                bestdelta = delta
                bestcolor = id
        # Found best
        return bestcolor
    else:
        # HSB/HSV calculations
        # Convert to HSV first (it will be used as a reference)
        # HSV is within 0..1,0..1,0..255
        rhsv = colorsys.rgb_to_hsv(*rgb)
        # Normalize to 0..1,0..1,0..1
        rhsv = (rhsv[0], rhsv[1], rhsv[2] / 255)
        # Set comparator
        bestdelta = 10
        bestcolor = 0
        # We have reference. Now scan through the colors
        for id in crange:
            color = Mpalette[id]
            phsv = colorsys.rgb_to_hsv(*color)
            phsv = (phsv[0], phsv[1], phsv[2] / 255)
            deltas = [abs(rhsv[c] - phsv[c]) for c in (0, 1, 2)]
            # Deltas gathered. Ponder some if needed
            if priority == 2: deltas[0] *= COLOR_MATCHING_PRIO_FACTOR
            if priority == 3:
                deltas[0] *= COLOR_MATCHING_PRIO_FACTOR
                deltas[2] *= COLOR_MATCHING_PRIO_FACTOR
            # Check how it scores
            delta = sum(deltas)
            if delta < bestdelta:
                bestdelta = delta
                bestcolor = id
        # By now 'bestcolor' is filled with index of the best matching color
        return bestcolor


def gradient32(rgb1, rgb2):
    '''
    Helper. Generate a smooth transition of length 32 where rgb1 is at postition 0 and rgb 2 at
    position 31 (end), and return a list of rgb's.
    '''
    delta = [rgb2[c] - rgb1[c] for c in (0, 1, 2)]
    step = [delta[c] / 31 for c in (0, 1, 2)]
    grad = [[rgb1[c] + light * step[c] for c in (0, 1, 2)] for light in range(32)]
    return grad


def homogenize(cmprange, sacrifice):
    '''
    Consider the color map in 'cmprange' and try to homogenize it by introducing new colors in
    place of 'sacrifice' which can obviously be replaced with the suitable ones in order to
    create as smooth 'cmprange' color map area as possible.
    Calculate ideal transitions from the first to the last color, then find the worst deviation,
    patch it with a new color, and repeat as long as there are colors to sacrifice.
    '''
    global Mpalette
    global Mcolormap

    # Initial homogenization of that range
    for col in cmprange:
        colrange = gradient32(Mpalette[Mcolormap[col][0]], Mpalette[Mcolormap[col][31]])
        colrange = [get_nearest_color(c) for c in colrange]
        Mcolormap[col] = colrange

    # Iterate and 'spend' sacrificial colors until none are left in 'sacrifice'
    while sacrifice:
        if DEBUG_MODE: print('Homogenization, colors left:', len(sacrifice))
        # Holder of the best candidates (initialization)
        bestdelta = -1
        bestcolor = [0, 0, 0]
        # Go through the ranges and get current deltas from the ideal transition
        for col in cmprange:
            colrange = gradient32(Mpalette[Mcolormap[col][0]], Mpalette[Mcolormap[col][31]])
            # Calculate deviation at each lightness
            for light in range(32):
                ldelta = [abs(Mpalette[Mcolormap[col][light]][c] - colrange[light][c])
                          for c in (0, 1, 2)]
                ldelta = sum(ldelta)
                # Check if it's the best candidate yet
                if ldelta > bestdelta:
                    # It is. Mark it
                    bestdelta = ldelta
                    bestcolor = colrange[light]
        # Analysis finished, the best color to take is in 'bestcolor'
        # Now assign it to one of the sacrificial colors and recalculate
        bestcolor = [norm255(bestcolor[c]) for c in (0, 1, 2)]
        Mpalette[sacrifice[0]] = copy.deepcopy(bestcolor)
        sacrifice = sacrifice[1:]
        # Recalculate now that we have a new color
        for col in cmprange:
            colrange = gradient32(Mpalette[Mcolormap[col][0]], Mpalette[Mcolormap[col][31]])
            colrange = [get_nearest_color(c) for c in colrange]
            Mcolormap[col] = colrange


def parse_ranges(input):
    '''Get a ranged-comma-delimited list of integers and split it to the corresponding
    numberic values. E.g. "0,4,5-8,9,11-13" leads to a list:
    [0,4,5,7,8,9,11,12,13]
    which is then returned.
    '''
    # Split to master tokens by comma
    input = input.split(',')
    # Set up a final collector
    items = []
    # Process each
    for token in input:
        if '-' in token:
            # A range
            first, tmp, last = token.partition('-')
            first = round(float(first))
            last = round(float(last))
            full = list(range(first, last + 1))
            items.extend(full)
        else:
            # Just a simple number
            items.append(round(float(token)))
    # Done, so just return the collector
    return items


def parse_pairs(input):
    '''Get multiple pairs of integers joined by '>', and further separated by commas, and
    return as a dictionary of first leading to second'''
    # Split to master tokens by comma
    input = input.split(',')
    # Set up a final collector
    items = dict()
    # Process each
    for token in input:
        first, tmp, last = token.partition('>')
        first = round(float(first))
        last = round(float(last))
        items[first] = last
    # Done, so just return the collector
    return items


def get_flashlight_range(level=50):
    '''
    Get 128 bytes of flashlight range added to the colormap upon saving. Vary it from
    normal (NARSHADA default) linear which is 50, to fully short-range 0, and to extreme
    long-range and bright 100.
    Approximate to range 0..1^x, where x varies from 0.1 over 1 to 10.
    '''
    # Quick shortcut if 0:
    if level == 0:
        return [31] * 128  # Full dark

    # Convert percentage to exponent 0.1->1->10 = 10^-1 -> 10^0 -> 10^1
    level = (level - 50) / 50
    exponential = 10**level  # Get 0.1 -> 1 -> 10
    # Create expo list
    lightlist = [(r / 127)**exponential for r in range(0, 128)]
    # Normalize to 0-31
    lightlist = [round(v * 31) for v in lightlist]
    return lightlist


def linear_homogenization(factor=5):
    '''
    Linear homogenization: detect the "starting" columns in the lightness map according to its
    changes in brightness, then from each of them, extrapolate the darker area next to them
    by stretching their values (keeping the darkest as it is, and stretching the lightest out)
    which should resemble the generic DF palettes more closely - hopefully.
    '''
    # First, need to find the "starters", i.e. each column where the brightness is higher than
    # in the previous one
    light = [sum([sum(Mpalette[Mcolormap[col][lt]]) for lt in range(32)]) for col in range(256)]
    leads = [e for e in range(1, 255) if light[e] > light[e - 1] and e >= 32]  # Section leaders
    if 32 not in leads: leads = [32] + leads  # Always begin from 32!
    # Get group ranges
    ranges = [(leads[k], leads[k + 1]) for k in range(len(leads) - 1)]
    ranges.append((leads[-1], 256))
    # Remove zero-lengths
    ranges = [e for e in ranges if e[1] - e[0] >= 2]
    # Ranges defined. Iterate through each
    wcmp = copy.deepcopy(Mcolormap)  # Work colormap
    for lead, trail in ranges:
        initcmp = wcmp[lead]  # Reference color column
        rangelen = trail - lead - 1
        stretch = (factor - 1) / rangelen
        for color in range(lead + 1, trail):
            prog = color - lead - 1  # Local incrementer
            locstretch = (prog + 1) * stretch + 1  # Size of this column
            locstep = 1 / locstretch  # Local incerement step at this column
            if DEBUG_MODE: print(color, prog, locstretch, locstep)
            # Increment per that step
            for light in range(32):
                lprog = light * locstep  # Current progression
                lprog = round(lprog)
                # Normalize
                if lprog < 0: lprog = 0
                if lprog > 31: lprog = 31
                # Find color and replace it
                tgtcol = initcmp[lprog]
                wcmp[color][light] = tgtcol
    # All done, return the work colormap
    return wcmp


# SUBLAYOUTS AND WINDOWS \2
# ===============================================================================================

# Modify color


def modify_color():
    '''Change the RGB values of the currently selected colors.'''
    global Mpalette
    curr, curg, curb = Mpalette[Mselcolor]
    # Set up the custom layout
    MODlay = [[sg.T('RGB channels')],
              [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                         trough_color='#770000', size=(50, 20), key='slider',
                         default_value=curr)],
              [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                         trough_color='#004400', size=(50, 20), key='slideg',
                         default_value=curg)],
              [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                         trough_color='#0000aa', size=(50, 20), key='slideb',
                         default_value=curb)],
              [sg.T(' ')],
              [sg.T('Exact RGB values:'), sg.Input(default_text='{0}, {1}, {2}'.
                                                   format(curr, curg, curb),
                                                   enable_events=True,
                                                   key='textrgb')],
              [sg.T(' ')],
              [sg.T('New color'),
               sg.Graph(canvas_size=(140, 40), background_color=gethexcol(curr, curg, curb),
                        graph_bottom_left=(0, 0), graph_top_right=(40, 40), key='preview',),
               sg.Graph(canvas_size=(140, 40), background_color=gethexcol(curr, curg, curb),
                        key='origcol', graph_bottom_left=(0, 0), graph_top_right=(40, 40)),
               sg.T('Original color')],
              [sg.T(' ')],
              [sg.Button('Revert', size=(10, 1)), sg.Button('Cancel', size=(10, 1)),
               sg.Button('OK', size=(10, 1), focus=True)],
              ]
    # Flip out the window
    ModifyColor = sg.Window(
        'Modify color',
        layout=MODlay,
        resizable=False,
        finalize=True,
        element_justification='center',
        font=(FONT_WINDOW, FONT_WINDOW_SIZE),
        modal=True,
        icon=ICONFILE,
    )
    # Event loop
    while True:
        action, keys = ModifyColor.read()

        if DEBUG_MODE: print(action, keys)

        # End or cancel
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            ModifyColor.close()
            break

        # Revert
        if action == 'Revert':
            ModifyColor['slider'].update(value=curr)
            ModifyColor['slideg'].update(value=curg)
            ModifyColor['slideb'].update(value=curb)
            ModifyColor['textrgb'].update(value='{0}, {1}, {2}'.format(curr, curg, curb))
            ModifyColor['preview'].update(background_color=gethexcol(curr, curg, curb))

        # Change slide values
        if action.startswith('slide'):
            # Update color and RGB text box
            nco = round(keys['slider']), round(keys['slideg']), round(keys['slideb'])
            ModifyColor['textrgb'].update(value='{0}, {1}, {2}'.format(*nco))
            ModifyColor['preview'].update(background_color=gethexcol(*nco))

        # Change text value
        if action == 'textrgb':
            nco = keys['textrgb']
            # Attempt parsing
            if not nco: continue
            nco = nco.split(',')
            if len(nco) != 3: continue  # We need EXACT 3 channels
            try:
                nco = [norm255(round(float(e))) for e in nco]
            except BaseException:
                continue
            # Passed all the hurdles, use this
            ModifyColor['slider'].update(value=nco[0])
            ModifyColor['slideg'].update(value=nco[1])
            ModifyColor['slideb'].update(value=nco[2])
            ModifyColor['preview'].update(background_color=gethexcol(*nco))

        # OK
        if action == 'OK':
            nco = keys['textrgb']
            # Attempt parsing
            nco = nco.split(',')
            try:
                fr, fg, fb = [norm255(round(float(e))) for e in nco]
            except BaseException:
                popup('Illegal color values entered! Please use R,G,B format.')
                continue
            # Passed all the hurdles, use this and finally apply it
            save_undo()
            Mpalette[Mselcolor] = [fr, fg, fb]
            ModifyColor.close()
            draw_all()
            break

# Open PAL and/or CMP


def open_files():
    global Mpalette
    global Mcolormap
    global Mfiles
    oplayout = [[sg.In(key='ppal'), sg.FileBrowse('Browse PAL', file_types=(('Palettes', '*.PAL'),),)],
                [sg.In(key='pcmp'), sg.FileBrowse('Browse CMP', file_types=(('Color maps', '*.CMP'),)), ],
                [sg.T('Hint: you can open only one file by leaving the other field blank.')],
                [sg.Button('Cancel', size=(10, 1)), sg.Button('Open', size=(10, 1), focus=True)],
                ]
    OpenFile = sg.Window('Open files', layout=oplayout,
                         resizable=False,
                         finalize=True,
                         element_justification='center',
                         font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                         modal=True,
                         icon=ICONFILE,
                         margins=(20, 20))
    # Event loop
    while True:
        action, keys = OpenFile.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            OpenFile.close()
            break

        # Open
        if action == 'Open':
            ppal = keys['ppal']
            pcmp = keys['pcmp']
            # Is there anything specified?
            if not ppal and not pcmp:
                popup('No files specified - nothing to do.')
                continue
            # There is something, let's check
            save_undo()
            if ppal:
                try:
                    lpal = load_palette(ppal)  # Give it a try
                    Mpalette = lpal
                    Mfiles[0] = ppal
                except BaseException:
                    popup('Error loading palette ' + ppal + '!')
            if pcmp:
                try:
                    lcmp = load_colormap(pcmp)  # Give it a try
                    Mcolormap = lcmp
                    Mfiles[1] = pcmp
                except BaseException:
                    popup('Error loading colormap ' + pcmp + '!')
            # Loaded, take care of the refreshes and closes
            OpenFile.close()
            draw_all()
            break

# Save file as


def save_files_as():
    oplayout = [[sg.In(key='ppal'), sg.FileSaveAs('Browse PAL',
                                                  file_types=(('Palettes', '*.PAL'),),),
                 sg.Checkbox('Exclude', key='cpal')],
                [sg.In(key='pcmp'), sg.FileSaveAs('Browse CMP',
                                                  file_types=(('Color maps', '*.CMP'),)),
                 sg.Checkbox('Exclude', key='ccmp')],
                [sg.T('\nFlashlight luminosity:')],
                [sg.T('Weaker, shorter range'),
                 sg.Slider(range=(0, 100), default_value=50, orientation='horizontal',
                           size=(60, None), key='mix'),
                 sg.T('Brighter, longer range')],
                [sg.T('\nSet the slidebar to 0 to disable the flashlight entirely. Keep it at 50 '
                      'for the flashlight like the one in the original missions.\n'), ],
                [sg.Button('Cancel', size=(24, 1)), sg.Button('Save', size=(24, 1), focus=True)], ]
    SaveAs = sg.Window('Save files as...', layout=oplayout,
                       resizable=False,
                       finalize=True,
                       element_justification='center',
                       font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                       modal=True,
                       icon=ICONFILE,
                       margins=(20, 20))
    # Preload existing names (if any)
    if Mfiles[0] != 'Untitled':
        SaveAs['ppal'].update(value=Mfiles[0])
    if Mfiles[1] != 'Untitled':
        SaveAs['pcmp'].update(value=Mfiles[1])

    # Event loop
    while True:
        action, keys = SaveAs.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            SaveAs.close()
            break

        # Save
        if action == 'Save':
            # Is there anything to save?
            if keys['cpal'] and keys['ccmp']:
                popup('Nothing to save. Please deselect at least one "Exclude" checkbox.')
                continue
            # Check what to save
            if not keys['cpal'] and keys['ppal']:
                # Save palette
                try:
                    save_palette(Mpalette, keys['ppal'])
                    Mfiles[0] = keys['ppal']
                except BaseException:
                    popup('Failed saving palette ' + keys['ppal'] + '!')
            if not keys['ccmp'] and keys['pcmp']:
                # Save colormap
                try:
                    save_colormap(Mcolormap, keys['pcmp'], flashlight=round(keys['mix']))
                    Mfiles[1] = keys['pcmp']
                except BaseException:
                    popup('Failed saving colormap ' + keys['pcmp'] + '!')
            # Close and leave
            SaveAs.close()
            update_title()
            break

# Apply tint


def apply_tint():

    def tintshow(pal):
        'Local show palette in the window.'
        Tint['preview'].erase()
        for col in range(256):
            gx = col % 16 * 16
            gy = col // 16 * 16
            Tint['preview'].draw_rectangle((gx, gy), (gx + 16, gy + 16), gethexcol(*pal[col]))

    global Mpalette

    cur = Mpalette[Mselcolor]
    TINTlayout = [[sg.T('Tinted color range:'), sg.In('0', size=(8, 1), key='rangefrom'),
                   sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
                   sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
                  [sg.T('')],
                  [sg.Frame('Tinting type', element_justification='left', layout=[
                      [sg.Radio('Tint (mix) colors by amount:  Min', group_id=1, key='t1',
                                enable_events=True),
                       sg.Slider(range=(0, 100), default_value=50, orientation='horizontal',
                                 disable_number_display=True, enable_events=True, key='mix'),
                       sg.T('Max')],
                      [sg.Radio('Add RGB to palette', group_id=1, key='t2', enable_events=True)],
                      [sg.Radio('Subtract RGB from palette', group_id=1, key='t3',
                                enable_events=True)],
                  ])],
                  [sg.T('')],
                  [sg.T('Tinting amounts (initially preset to selected color):')],
                  [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                             trough_color='#770000', size=(50, 20), key='slider',
                             default_value=cur[0])],
                  [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                             trough_color='#004400', size=(50, 20), key='slideg',
                             default_value=cur[1])],
                  [sg.Slider(range=(0, 255), orientation='horizontal', enable_events=True,
                             trough_color='#0000aa', size=(50, 20), key='slideb',
                             default_value=cur[2])],
                  [sg.T('\nPreview of the entire palette (disregards range above):')],
                  [sg.Graph(canvas_size=(256, 256), background_color='#000000',
                            graph_bottom_left=(0, 256), graph_top_right=(256, 0), key='preview',), ],
                  [sg.T('')],
                  [sg.Button('Cancel', size=(15, 1)), sg.Button('Tint', size=(15, 1), focus=True)],
                  ]
    Tint = sg.Window('Tint palette', layout=TINTlayout,
                     resizable=False,
                     finalize=True,
                     element_justification='center',
                     font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                     modal=True,
                     icon=ICONFILE,
                     margins=(20, 20))
    # Preselect default tint
    Tint['t1'].update(value=True)
    # Pre-calculate first palette
    temppal = calc_tint(operation=0, targetrgb=cur, mix=50)
    tintshow(temppal)
    # Event loop
    while True:
        action, keys = Tint.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Tint.close()
            break

        # Color range quick presets
        if action == '0255':
            Tint['rangefrom'].update(value='0')
            Tint['rangeto'].update(value='255')
        elif action == '32255':
            Tint['rangefrom'].update(value='32')
            Tint['rangeto'].update(value='255')

        # Update controls and recalculate temporary palette
        # Get operation
        if keys['t1']: op = 0
        elif keys['t2']: op = 1
        elif keys['t3']: op = 2
        # Get RGB targets
        trgb = (norm255(keys['slider']), norm255(keys['slideg']), norm255(keys['slideb']))
        # Calculate and show
        temppal = calc_tint(operation=op, targetrgb=trgb, mix=round(keys['mix']))
        tintshow(temppal)

        # OK, apply
        if action == 'Tint':
            # Determine where
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue

            save_undo()
            for clindex in applyrange:
                Mpalette[clindex] = temppal[clindex]
            Tint.close()
            draw_all()

# Palette checks


def redundancy_consistency_check():
    duplicates, unused = palette_check()
    # Get assembling text
    if not duplicates and not unused:
        # Everything OK!
        popup('The Force is strong with this palette! It has no duplicates nor any unused colors.')
        return

    # Some problems found
    text = []
    # Take care of duplicates
    if duplicates:
        if len(duplicates) > 300:  # Check if there are too many
            text.append('This palette has more than 300 duplicate pairs of colors.')
        else:
            # Not - so display them
            duplicates = [str(a) + '=' + str(b) for a, b in duplicates]
            text.append('The following pairs of colors are identical:')
            text.append('')
            text.append(', '.join(duplicates))
        comment = 'Note that these pairs may actually have slightly differing RGB values, ' +\
            'but due to 6-bit reduction used by Dark Forces, some will turn out to ' +\
            'be identical in the game. E.g. values 203 and 201 both round to 200.'
        text.append('')
        text.append(comment)
        text.append('')
        text.append('')

    # Take care of unused colors
    if unused:
        comment = 'The following colors are unused, i.e. they do not ' +\
            'appear in the color map even under their own index number at full light:'
        text.append(comment)
        text.append('')
        unused = [str(e) for e in unused]
        text.append(', '.join(unused))
        text.append('')
        text.append('(However, be aware that the Dark Forces engine always shows the original ' +
                    'palette color at full brightness (31), regardless of the colormap ' +
                    'value assigned. So, changing these colors may lead to unexpected ' +
                    'or unwanted results!)')
        text.append('')

    # Assemble together
    final = '\n'.join(text)

    # Show in a special popup where the content can be selected and copied from
    sg.popup_scrolled(final,
                      title='Consistency & redundancy checks',
                      icon=ICONFILE,
                      modal=True,
                      size=(120, 20))

# Palette adjustments - brightness, contrast, etc.


def palette_adjustments():
    global Mpalette

    def adjshow(pal):
        'Local show palette in the window.'
        Adjust['preview'].erase()
        for col in range(256):
            gx = col % 16 * 16
            gy = col // 16 * 16
            Adjust['preview'].draw_rectangle((gx, gy), (gx + 16, gy + 16), gethexcol(*pal[col]))

    ADJlay = [[sg.T('Tinted color range:'), sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('')],
              [sg.T('Brightness')],
              [sg.Slider(range=(-255, 255), orientation='horizontal', enable_events=True,
                         size=(50, 20), key='br', default_value=0)],
              [sg.T('\nContrast')],
              [sg.Slider(range=(-50, 50), orientation='horizontal', enable_events=True,
                         size=(50, 20), key='co', default_value=0)],
              [sg.T('\nGamma')],
              [sg.Slider(range=(-50, 50), orientation='horizontal', enable_events=True,
                         size=(50, 20), key='ga', default_value=0)],
              [sg.T('\nSaturation')],
              [sg.Slider(range=(-50, 50), orientation='horizontal', enable_events=True,
                         size=(50, 20), key='sa', default_value=0)],
              [sg.T('\nPreview of the entire palette (disregards range above):')],
              [sg.Graph(canvas_size=(256, 256), background_color='#000000',
                        graph_bottom_left=(0, 256), graph_top_right=(256, 0), key='preview',), ],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Apply', size=(15, 1), focus=True)],
              ]
    Adjust = sg.Window('Palette adjustments', layout=ADJlay,
                       resizable=False,
                       finalize=True,
                       element_justification='center',
                       font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                       modal=True,
                       icon=ICONFILE,
                       margins=(20, 20))
    adjshow(Mpalette)  # First show just existing, unchanged palette
    temppal = copy.deepcopy(Mpalette)  # The one that will be worked upon
    # Event loop
    while True:
        action, keys = Adjust.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Adjust.close()
            break

        # Color range quick presets
        if action == '0255':
            Adjust['rangefrom'].update(value='0')
            Adjust['rangeto'].update(value='255')
        elif action == '32255':
            Adjust['rangefrom'].update(value='32')
            Adjust['rangeto'].update(value='255')

        # Update preview and recalculate
        temppal = calc_adjustments(keys['br'], keys['co'], keys['ga'], keys['sa'])
        adjshow(temppal)

        # OK, apply
        if action == 'Apply':
            # Determine where
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue

            save_undo()
            for clindex in applyrange:
                Mpalette[clindex] = temppal[clindex]
            Adjust.close()
            draw_all()

# Auto gradient between 2 palette colors


def auto_gradient():
    global Mpalette
    WINlay = [[sg.T('Please enter the color indexes to bridge with a gradient, separated by comma:')],
              [sg.In(str(Mselcolor) + ', 255', key='ind')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Generate', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Auto-gradient',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK, apply
        if action == 'Generate':
            # Get range
            try:
                fcol, tmp, lcol = keys['ind'].partition(',')
                fcol = round(float(fcol))
                lcol = round(float(lcol))
                fcol = norm255(fcol)
                lcol = norm255(lcol)
            except BaseException:
                popup('Illegal color range. Please use format Index1,Index2.')
                continue
            # Perform gradient
            save_undo()
            Mpalette = generate_gradient(fcol, lcol)
            # Clean up
            Win.close()
            draw_all()


# Import paletted bitmap
def import_paletted_img():
    global Mpalette

    WINlay = [[sg.T('Pick a paletted image to extract the palette from:')],
              [sg.In('', key='file'),
               sg.FileBrowse('Browse', file_types=(('All', '*.*'),),)],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)),
               sg.Button('Load palette', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Import palette from image',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK/apply
        if action == 'Load palette':
            # Try to extract palette
            try:
                img = pil.open(keys['file'])
                pal = img.getpalette()
                wpal = []  # Temporary work palette
                for coli in range(0, len(pal), 3):
                    wpal.append([pal[coli], pal[coli + 1], pal[coli + 2]])
            except BaseException:
                popup('Failed loading the given image. (It may not exist or not be paletted.)')
                continue
            # Loaded - apply now
            save_undo()
            for id, clr in enumerate(wpal):
                Mpalette[id] = clr
            # Cleanup
            Win.close()
            draw_all()

# Merge current palette with another one


def merge_palette():
    global Mpalette

    WINlay = [[sg.T('Pick a palette .PAL file to merge the current one with:')],
              [sg.In('', key='file'),
               sg.FileBrowse('Browse', file_types=(('All', '*.*'),),)],
              [sg.T('')],
              [sg.T('Color range to be merged from the file:'),
               sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('Color index offset (positive or negative, 0 keeps in place):'),
               sg.In('0', size=(8, 1), key='indoffset')],
              [sg.Checkbox('Mix colors instead of replacing them', key='mixmet'), ],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Merge', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Merge palettes',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK/apply
        if action == 'Merge':
            # Determine where
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
                applyoffset = int(keys['indoffset'])
            except BaseException:
                popup('Please enter a valid color range and offset!')
                continue
            # Load file
            try:
                mergedpal = load_palette(keys['file'])
            except BaseException:
                popup('Failed loading palette ' + keys['file'] + '!')
                continue
            # Successful on both operations - apply now
            save_undo()
            for icol in applyrange:
                colposition = norm255(icol + applyoffset)
                # See whether replacing or mixing
                if not keys['mixmet']:
                    # Just replace
                    Mpalette[colposition] = mergedpal[icol]
                else:
                    # Mix with existing
                    newcol = [norm255((Mpalette[colposition][c] + mergedpal[icol][c])
                                      // 2) for c in (0, 1, 2)]
                    Mpalette[colposition] = newcol
            # Clean up
            Win.close()
            draw_all()


# Copy colors
def copy_color():
    global Mpalette

    WINlay = [[sg.T('Enter the indexes to copy the current color to, separated by comma:')],
              [sg.In('', key='clrs'), ],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Copy', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Copy color',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
# Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        if action == 'Copy':
            # Try to determine values
            try:
                vals = keys['clrs'].split(',')
                vals = [norm255(float(v)) for v in vals]
            except BaseException:
                popup('Wrong entries in the indexes list. Use e.g. 4,6,8,23')
                continue
            # Perform copies
            save_undo()
            for cind in vals:
                Mpalette[cind] = copy.deepcopy(Mpalette[Mselcolor])
            # Cleanup
            Win.close()
            draw_all()


# Auto-calculate color map
def auto_calculate_map():
    global Mcolormap

    WINlay = [
        [sg.T('AFFECTED COLOR MAP AREA')],
        [sg.T('Palette range to calculate:'), sg.In('0', size=(8, 1), key='rangefrom'),
         sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
         sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
        [sg.T('Color map light range to calculate:'), sg.In('0', size=(8, 1), key='lightfrom'),
         sg.T('to'), sg.In('31', size=(8, 1), key='lightto'), ],
        [sg.T('\nCALCULATION PARAMETERS')],
        [sg.T('Projected light range:'), sg.In('0', size=(8, 1), key='projfrom'),
         sg.T('to'), sg.In('31', size=(8, 1), key='projto'), ],
        [sg.Frame('Calculation method', element_justification='left', layout=[
            [sg.Radio('Linear from full black to maximum brightness (Typical)', group_id=0, key='met1')],
            [sg.Radio('From selected color to maximum brightness (Distance haze effect)',
                      group_id=0, key='met2')],
            [sg.Radio('Use selected color as a color limiter (Water/Alert light effect)',
                      group_id=0, key='met3')],
            [sg.Radio('Rain trickle algorithm (Favours map homogeneity, ignores light ranges)',
                      group_id=0, key='met4')],
            [sg.Radio('Hue shift (Ignores light ranges, light-change psychodelics)',
                      group_id=0, key='met5')],
        ])],
        [sg.Frame('Color choice', element_justification='left', layout=[
            [sg.Radio('Find the color nearest to the calculated one (RGB)', group_id=1, key='ch0')],
            [sg.Radio('Find the color nearest to the calculated one (HSB)', group_id=1, key='ch1')],
            [sg.Radio('Prioritize hue', group_id=1, key='ch2')],
            [sg.Radio('Deprioritize saturation', group_id=1, key='ch3')],

        ])], [sg.Checkbox('Reverse search order', key='revorder')],
        [sg.Frame('Handling special colors', element_justification='left', layout=[
            [sg.Checkbox('Force maximum light for luminescent colors (0-23)', key='maxlt')],
            [sg.Checkbox('Ignore special lights (24-31)', key='ignorespcl')],
        ])],
        [sg.T('')],
        [sg.Button('Cancel', size=(15, 1)), sg.Button('Hints', size=(15, 1)),
         sg.Button('Calculate', size=(15, 1), focus=True)],
    ]
    Win = sg.Window('Auto-calculate color map',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Set default values
    Win['maxlt'].update(value=True)
    Win['ignorespcl'].update(value=True)
    Win['met1'].update(value=True)
    Win['ch0'].update(value=True)
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Hints
        if action == 'Hints':
            popup(cmconst.AUTOMAPHINTS)

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # Apply/OK
        if action == 'Calculate':
            # Determine where
            try:
                applycolorrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            try:
                applylightrange = range(int(keys['lightfrom']), int(keys['lightto']) + 1)
            except BaseException:
                popup('Please enter a valid light level range!')
                continue
            # Determine lighting range
            try:
                startlight = round(float(keys['projfrom']))
                endlight = round(float(keys['projto']))
            except BaseException:
                popup('Please enter valid projected light levels!')
                continue
            # Determine method
            if keys['met1']: method = 1
            elif keys['met2']: method = 2
            elif keys['met3']: method = 3
            elif keys['met4']: method = 4
            else: method = 5
            # Determine choice priority
            if keys['ch0']: priority = 0
            elif keys['ch1']: priority = 1
            elif keys['ch2']: priority = 2
            else: priority = 3
            # Boolean parameters
            keepluminescent = keys['maxlt']
            ignore2431 = keys['ignorespcl']
            reverseorder = keys['revorder']
            # Parameters prepared by this point, do the calculation
            Win.close()
            try:
                newmap = calc_color_map(applycolorrange,
                                        applylightrange,
                                        (startlight, endlight),
                                        method,
                                        priority,
                                        keepluminescent,
                                        ignore2431,
                                        reverseorder,
                                        )
            except BaseException:
                popup('Sorry, failed the calculation with current parameters.')
                break
            save_undo()
            # All ready, now replace with the new one
            Mcolormap = copy.deepcopy(newmap)
            draw_all()

# Merge colormap (replace with)


def merge_colormap():
    global Mcolormap

    WINlay = [[sg.T('Pick a color map .CMP file to replace the current one with:')],
              [sg.In('', key='file'),
               sg.FileBrowse('Browse', file_types=(('All', '*.*'),),)],
              [sg.T('')],
              [sg.T('Color range to be merged from the file:'),
               sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('Color index offset (positive or negative, 0 keeps in place):'),
               sg.In('0', size=(8, 1), key='indoffset')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Replace', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Replace color map',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK/apply
        if action == 'Replace':
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
                applyoffset = int(keys['indoffset'])
            except BaseException:
                popup('Please enter a valid color range and offset!')
                continue
            # Load file
            try:
                mergedcmp = load_colormap(keys['file'])
            except BaseException:
                popup('Failed loading colormap ' + keys['file'] + '!')
                continue
            # Successful on both operations - apply now
            save_undo()
            for icol in applyrange:
                colposition = norm255(icol + applyoffset)
                Mcolormap[colposition] = mergedcmp[icol]
            # Clean up
            Win.close()
            draw_all()


# Preview a look of a paletted bitmap with this palette and colormap
def preview_pal_bmp():
    WINlay = [[sg.T('Paletted .BMP filename:')],
              [sg.In('', key='file', enable_events=True),
               sg.FileBrowse('Browse', file_types=(('Bitmaps', '*.BMP'),),)],
              [sg.T('')],
              [sg.Graph(canvas_size=(256, 256), graph_bottom_left=(0, 256),
                        graph_top_right=(256, 0), key='prevgraph')],
              [sg.T('\nPreviewed light level:')],
              [sg.Slider(range=(0, 31), default_value=31, key='ltslider', orientation='h',
                         enable_events=True)],
              [sg.T('')],
              [sg.Button('Close', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Preview a bitmap with current palette and map',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Temporarily create a tuple map for faster operation
    tempmap = copy.deepcopy(Mcolormap)
    tempmap = [tuple(e) for e in tempmap]
    tempmap = tuple(tempmap)

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Close':
            Win.close()
            break

        # OK/Apply
        if action == 'ltslider' or action == 'file':
            if not keys['file']: continue  # No file entered yet
            # Load the image first
            try:
                img = pil.open(keys['file'])
                px = img.load()
                loffset = 128 - img.size[0] // 2
                toffset = 128 - img.size[1] // 2
            except BaseException:
                continue
            # Check if paletted indeed!
            if not isinstance(px[0, 0], int):
                popup('Not a valid paletted image!')
                continue
            # Iterate over "pixelation"
            Win['prevgraph'].erase()
            try:
                reflight = round(int(keys['ltslider']))
                for nx in range(img.size[0]):
                    for ny in range(img.size[1]):
                        if nx < 256 and ny < 256:
                            # Copy pixels
                            refcol = px[nx, ny]
                            loccol = tempmap[refcol][reflight]
                            Win['prevgraph'].draw_point((loffset + nx, toffset + ny), size=1,
                                                        color=gethexcol(*Mpalette[loccol]))
            except BaseException:
                popup('Failed calculating lights range!')
                continue


# Solve uneven segments of color maps
def solve_unevennesses():
    global Mcolormap
    global Mpalette

    WINlay = [[sg.T('Color map range to be solved:'), sg.In('192', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('207', size=(8, 1), key='rangeto'), ],
              [sg.T('\nPalette colors to "sacrifice" (comma delimited, optionally with ranges '
                    'separated by a dash):')],
              [sg.In('', size=(75, 1), key='sacrifice')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Hints', size=(15, 1)),
               sg.Button('Solve', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Solve uneven colormap transitions',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Hints
        if action == 'Hints':
            popup(cmconst.AUTOHOMOGENIZEHINTS)
            continue

        # OK, apply/solve
        if action == 'Solve':
            # Collect 'numbering'
            try:
                applyrange = list(range(int(keys['rangefrom']), int(keys['rangeto']) + 1))
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            try:
                sacrifice = parse_ranges(keys['sacrifice'])
            except BaseException:
                popup('Enter a correct list of color(s) to be sacrificed.')
                continue
            # Collected both. Exclude any colors within that very range
            sacrifice = [e for e in sacrifice if e not in applyrange]
            if not sacrifice:
                popup('Sacrificed colors cannot all be in the considered range.')
                continue
            # Collected all the necessary info. Try calculating
            save_undo()
            homogenize(applyrange, sacrifice)
            Win.close()
            draw_all()

# Sort colors in palette


def sort_palette():
    global Mpalette
    global Mcolormap

    WINlay = [[sg.T('Color range to be sorted:'), sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.Frame('Sorting criterion', element_justification='left', layout=[
                  [sg.Radio('From darkest to brightest', group_id=1, key='t1')],
                  [sg.Radio('By hue, starting with red', group_id=1, key='t2')],
                  [sg.Radio('From least to most saturated', group_id=1, key='t3')],
              ])],
              [sg.Checkbox('Reverse order', key='revorder')],
              [sg.Checkbox('Update colormap to reflect changes', key='updcol')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Sort', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Sort palette',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Preset
    Win['t1'].update(value=True)
    Win['updcol'].update(value=True)
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK/Apply
        if action == 'Sort':
            # Define range
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            # Build a range of [(r,g,b),colorindex]
            sorter = [(Mpalette[c], c) for c in applyrange]
            # Perform sort based on the method
            if keys['t1']:
                sorter.sort(key=lambda c: sum(c[0]), reverse=keys['revorder'])
            elif keys['t2']:
                sorter.sort(key=lambda c: colorsys.rgb_to_hsv(*c[0])[0], reverse=keys['revorder'])
            elif keys['t3']:
                sorter.sort(key=lambda c: colorsys.rgb_to_hsv(*c[0])[1], reverse=keys['revorder'])

            save_undo()
            # Apply to the original palette
            replacers = dict()  # Aggregator dict for colormap processing later
            for id, color in enumerate(sorter):
                Mpalette[id + applyrange[0]] = sorter[id][0]
                replacers[sorter[id][1]] = id + applyrange[0]
            # Apply to color map - iterate over all indexes
            if keys['updcol']:
                for color in range(0, 256):
                    for light in range(0, 32):
                        if Mcolormap[color][light] in replacers.keys():
                            Mcolormap[color][light] = copy.copy(replacers[Mcolormap[color][light]])
            # Clean up
            Win.close()
            draw_all()


# Show scatter plot of palette coverage
def coverage_scatter():
    WINlay = [[sg.T('Hue vs Brightness:')],
              [sg.Graph(canvas_size=(257, 257), background_color='#000000', key='scatter1',
                        graph_bottom_left=(0, 257), graph_top_right=(257, 0), enable_events=True)],
              [sg.T('')],
              [sg.T('Hue vs Saturation:')],
              [sg.Graph(canvas_size=(257, 257), background_color='#000000', key='scatter2',
                        graph_bottom_left=(0, 257), graph_top_right=(257, 0), enable_events=True)],
              [sg.T('')],
              [sg.Button('Close', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Color coverage scatter',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Perform drawing
    # Initial data first
    for n in range(256):
        # Legend
        Win['scatter1'].draw_point((n + 1, 0),
                                   size=1, color=gethexcol(*colorsys.hsv_to_rgb(n / 256, 1, 255)))
        Win['scatter1'].draw_point((0, n + 1),
                                   size=1, color=gethexcol(n, n, n))
        Win['scatter2'].draw_point((n + 1, 0),
                                   size=1, color=gethexcol(*colorsys.hsv_to_rgb(n / 256, 1, 255)))
        Win['scatter2'].draw_point((0, n + 1),
                                   size=1, color=gethexcol(255 - n, 255 - n, 255))

        # Actual color
        h, s, v = colorsys.rgb_to_hsv(*Mpalette[n])
        h = norm255(h * 255)
        s = norm255(s * 255)
        v = norm255(v)
        Win['scatter1'].draw_point((1 + h, 1 + v), size=4, color=gethexcol(*Mpalette[n]))
        Win['scatter2'].draw_point((1 + h, 1 + s), size=4, color=gethexcol(*Mpalette[n]))
        Win['scatter1'].draw_point((1 + h, 1 + v), size=1, color='#FFFFFF')
        Win['scatter2'].draw_point((1 + h, 1 + s), size=1, color='#FFFFFF')

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Close':
            Win.close()
            break

# Mass replace colors in the colormap according to manual rules


def mass_replace_colormap():
    global Mcolormap

    WINlay = [[sg.T('Affected color map range:'), sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('\n"From" and "To" color indexes separated by > '
                    ' (Separate multiple pairs with comma, e.g. 0>20, 30>50):')],
              [sg.In('', key='pairs', size=(75, 1))],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Replace', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Mass replace colors in the map',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK, apply
        if action == 'Replace':
            # Define range
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            # Then get the parsed dictionary
            try:
                replacers = parse_pairs(keys['pairs'])
            except BaseException:
                popup('Please enter a correct pair list A>B,C>D,E>F, etc.')
                continue
            if not replacers:
                popup('Please enter at least one value.')
                continue
            # Collected numbers, let's replace
            save_undo()
            # Iterate over all colormap
            for colind in applyrange:
                for light in range(0, 32):
                    if Mcolormap[colind][light] in replacers.keys():
                        Mcolormap[colind][light] = norm255(replacers[Mcolormap[colind][light]])
            # Done, refresh and clean up
            Win.close()
            draw_all()

# Load CMP from approximated image


def load_cmp_from_image():
    global Mcolormap

    WINlay = [[sg.T('Pick an image to load and convert to a color map:')],
              [sg.In(key='pcmp'), sg.FileBrowse('Browse', file_types=(('Images', '*.*'),)), ],

              [sg.T('(Images that do not have dimensions 256x32 px will be rescaled accordingly.)')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Load', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Load color map from image',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK/Apply
        if action == 'Load':
            # Attempt loading image first
            try:
                img = pil.open(keys['pcmp'])
                px = img.load()
            except BaseException:
                popup('Please enter a correct filename and make sure it is a valid image.')
                continue
            # Calculate steps horizontal/vertical
            stepx = img.size[0] / 256
            stepy = img.size[1] / 32
            # Iterate
            save_undo()
            for ky in range(32):
                locy = int(ky * stepy)  # Current Y location
                for kx in range(256):
                    locx = int(kx * stepx)  # Current X location
                    lpix = px[locx, locy]
                    replacer = get_nearest_color(lpix)
                    Mcolormap[kx][ky] = replacer
            # Done, clean up
            Win.close()
            draw_all()
            break

# Export CMP to image


def export_cmp_to_image():

    WINlay = [[sg.T('File to export the color map to:')],
              [sg.In(key='expc'), sg.FileSaveAs('Browse',
                                                file_types=(('PNG image', '*.PNG'),),), ],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Export', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Export color map to image',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK/apply
        if action == 'Export':
            img = pil.new('RGB', (256, 32), (0, 0, 0))
            px = img.load()
            # Iterate through all
            for ccol in range(0, 256):
                for light in range(0, 32):
                    px[ccol, light] = tuple(Mpalette[Mcolormap[ccol][light]])
            # Done. Attempt saving
            try:
                img.save(keys['expc'])
            except BaseException:
                popup('Please enter a correct export filename and make sure you have got'
                      'permissions to save to that location.')
            Win.close()
            break

# Progressive saturation, i.e. one depending on lightness


def prog_saturation():
    global Mpalette

    WINlay = [[sg.T('Affected palette range:'), sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('\nThis feature reduces saturation for dark colors, which avoids the '
                    + 'problem of dark areas in the game looking unnaturally lively.')],
              [sg.T('\nAmount:'),
               sg.Slider(range=(0, 100), orientation='horizontal', size=(50, 20),
                         key='slider', default_value=50)],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Replace', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Saturation depending on brightness',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK, apply
        if action == 'Replace':
            # Define range
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            # Range defined. Let's calculate
            multip = 1 - (float(keys['slider']) / 100)
            save_undo()
            for col in applyrange:
                # First get HSV
                h, s, v = colorsys.rgb_to_hsv(*Mpalette[col])
                maxsat = v / 63 * multip  # Get maximum saturation for this brightness
                if s > maxsat:  # Indeed, we need to change it
                    s = maxsat
                    newcol = colorsys.hsv_to_rgb(h, s, v)
                    newcol = [norm255(newcol[c]) for c in (0, 1, 2)]  # Calculate new
                    Mpalette[col] = copy.deepcopy(newcol)
            # Clean up
            Win.close()
            draw_all()

# Change theme


def change_theme():
    # Prefill values
    global COLORTHEME
    locthemes = sorted(cmconst.COLORTHEMES.keys())

    WINlay = [[sg.T('Pick a new theme:')],
              [sg.Combo(locthemes, default_value=COLORTHEME, readonly=True, key='combox')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Apply', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Change theme',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK/apply
        if action == 'Apply':
            # Check content
            if keys['combox'] not in cmconst.COLORTHEMES.keys(): continue
            COLORTHEME = keys['combox']
            Win.close()
            Main.close()
            init_Main_window()


# Homogenize map
def lin_homogenize():
    global Mcolormap

    WINlay = [[sg.T('Homogenization falloff factor:')],
              [sg.Slider(range=(2, 12), orientation='horizontal', size=(50, 20), key='slider',
                         default_value=6)],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Homogenize', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Linear color map homogenization',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK, apply
        if action == 'Homogenize':
            # Range defined. Let's calculate
            save_undo()
            try:
                newcmp = linear_homogenization(keys['slider'])
                Mcolormap = copy.deepcopy(newcmp)
            except BaseException:
                popup('Failed to calculate! Please readjust the edge light values.')
                continue
            # Clean up
            Win.close()
            draw_all()

# Unbalance the map colors acccording to a given curve


def unbalance():
    global Mcolormap

    WINlay = [[sg.T('Affected color map range:'), sg.In('0', size=(8, 1), key='rangefrom'),
               sg.T('to'), sg.In('255', size=(8, 1), key='rangeto'),
               sg.Button('0-255', key='0255'), sg.Button('32-255', key='32255')],
              [sg.T('')],
              [sg.T('Disbalance pivot')],
              [sg.T('Brighter'),
               sg.Slider(range=(-10, 10), orientation='horizontal', enable_events=True,
                         size=(50, 20), key='slider',
                         default_value=0),
               sg.T('Darker')],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)), sg.Button('Apply', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Unbalance the map',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))

    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # Color range quick presets
        if action == '0255':
            Win['rangefrom'].update(value='0')
            Win['rangeto'].update(value='255')
        elif action == '32255':
            Win['rangefrom'].update(value='32')
            Win['rangeto'].update(value='255')

        # OK, apply
        if action == 'Apply':
            # Check applicable range first
            try:
                applyrange = range(int(keys['rangefrom']), int(keys['rangeto']) + 1)
            except BaseException:
                popup('Please enter a valid color range!')
                continue
            save_undo()
            # Check the curvature
            expo = keys['slider']  # The exponential to use
            if expo == 0:
                # No change! Just return
                Win.close()
                break
            # Otherwise, there is some curvature
            # Normalize and align the exponential
            if expo > 1: expo = 1 + expo / 4
            else: expo = 1 / (1 + abs(expo) / 4)
            # Apply per each color in the map
            for col in applyrange:
                # Define temporary map
                localmap = []
                # Range over lights and calculate
                for light in range(32):
                    locl = light / 31
                    locl = locl**expo
                    locl = round(locl * 31)
                    localmap.append(Mcolormap[col][locl])
                Mcolormap[col] = copy.deepcopy(localmap)
            Win.close()
            draw_all()

# Make selected color luminous in the color map


def make_sel_luminous():
    global Mcolormap
    save_undo()
    for light in range(0, 32):
        Mcolormap[Mselcolor][light] = Mselcolor
    draw_all()

# Export the palette to a paletted image 256x1 px


def export_paletted_img():

    WINlay = [[sg.T('File to export the palette to:')],
              [sg.In(key='oimg'), sg.FileSaveAs('Browse BMP',
                                                file_types=(('Bitmaps', '*.BMP'),),), ],
              [sg.T('')],
              [sg.Button('Cancel', size=(15, 1)),
               sg.Button('Export', size=(15, 1), focus=True)],
              ]
    Win = sg.Window('Export palette to image',
                    layout=WINlay,
                    resizable=False,
                    finalize=True,
                    element_justification='center',
                    font=(FONT_WINDOW, FONT_WINDOW_SIZE),
                    modal=True,
                    icon=ICONFILE,
                    margins=(20, 20))
    # Event loop
    while True:
        action, keys = Win.read()
        if DEBUG_MODE: print(action, keys)

        # Cancel/close
        if action == sg.WIN_CLOSED or action == 'Exit':
            break
        if action == 'Cancel':
            Win.close()
            break

        # OK/apply
        if action == 'Export':
            # Try to set palette in a new image
            try:
                img = pil.new('P', (256, 1), 0)
                px = img.load()
                temppal = []  # Collector
                for x in range(256):
                    px[x, 0] = x
                    temppal.extend(Mpalette[x])
                img.putpalette(temppal)
                img.save(keys['oimg'])
            except BaseException:
                popup('Failed saving image. (May be related to insufficient write permissions.)')
                continue
            # Cleanup
            Win.close()
            draw_all()


#########################
# GENERAL LAYOUT START \3
# ===============================================================================================
# LOCAL INITIALIZATIONS
# Palette and map master initializations
Mpalette = [(0, 0, 0) for tmp in range(256)]
Mcolormap = [[0 for tmp2 in range(32)] for tmp in range(256)]
# Currently opened files [palfile, cmpfile] - by default just untitled
Mfiles = ['Untitled', 'Untitled']
Mselcolor = 0  # Currently selected color
Morigs = [entry[1] for entry in cmconst.ORIGS]
Mundo = []  # Master Undo register of (palette,colormap) entries
Mshowselinmap = True  # Show selected color in colormap?

# GUI

# Load SECBASE by default
Mpalette = copy.deepcopy(cmconst.ORIGS[0][2])
Mcolormap = copy.deepcopy(cmconst.ORIGS[0][3])
init_Main_window()

# Event loop
while True:
    action, keys = Main.read()
    if DEBUG_MODE: print(action, keys)

    # MASTER PARSE AND CHECK

    # End program, break loop
    if action == sg.WIN_CLOSED or action == 'Exit':
        break

    # MENU

    # Menu: New
    if action == 'New (N)' or action == 'n':
        save_undo()
        Mpalette = [(0, 0, 0) for tmp in range(256)]
        Mcolormap = [[tmp for tmp2 in range(32)] for tmp in range(256)]
        Mfiles = ['Untitled', 'Untitled']
        draw_all()

    # Menu: Open
    if action == 'Open... (O)' or action == 'o':
        open_files()

    # Menu: Save
    if action == 'Save (S)' or action == 's':
        # Check if we have valid names
        if Mfiles[0] == 'Untitled' or Mfiles[1] == 'Untitled':
            # Not, so use Save As workflow instead
            save_files_as()
            continue
        # Attempt saving automatically. Save palette
        try:
            save_palette(Mpalette, Mfiles[0])
        except BaseException:
            popup('Failed saving palette ' + Mfiles[0] + '!')
        # Save colormap
        try:
            save_colormap(Mcolormap, Mfiles[1])
        except BaseException:
            popup('Failed saving colormap ' + Mfiles[1] + '!')

    # Menu: Save as
    if action == 'Save as... (A)' or action == 'a':
        save_files_as()

    # Menu: Quit
    if action == 'Quit' or action == 'q':
        Main.close()  # To properly clean behind itself
        break

    # Menu: Undo
    if action == 'Undo (U)' or action == 'u':
        undo()

    # Menu: Toggle swatch borders
    if action == 'Toggle swatch borders (B)' or action == 'b':
        SWATCH_BORDER = abs(SWATCH_BORDER - 1)
        draw_all()

    # Menu: Reverse light orientation
    if action == 'Reverse light orientation (-)' or action == '-':
        # Check and act accordingly
        REVERSE_LIGHT = not REVERSE_LIGHT
        if REVERSE_LIGHT:
            Main['colormap'].change_coordinates((0, 0), (1024, 256))
        else:
            Main['colormap'].change_coordinates((0, 256), (1024, 0))
        draw_all()

    # Menu: resize window
    if action == 'Window size... (Z)' or action == 'z':
        zr = sg.popup_get_text('Enter the new zoom level percentage (75-300):',
                               'Set window size', '100')
        if not zr: continue  # Nothing entered or cancelled
        try:
            zr = round(float(zr))
            SCALE = round(zr * 32 / 100)
        except BaseException:
            popup('Wrong zoom value!')
            continue
        # Validate and set
        if SCALE < 24: SCALE = 24
        if SCALE > 96: SCALE = 96
        Main.close()
        init_Main_window()
    # Resize hotkeys
    if action == ']':
        SCALE = round(SCALE * 1.2)
        if SCALE > 96: SCALE = 96
        Main.close()
        init_Main_window()
    if action == '[':
        SCALE = round(SCALE / 1.2)
        if SCALE < 24: SCALE = 24
        Main.close()
        init_Main_window()

    # Menu: Window theme
    if action == 'Window theme...':
        change_theme()

    # Menu: Toggle selected color
    if action == 'Toggle selected color (\\)' or action == '\\':
        Mshowselinmap = not Mshowselinmap
        draw_all()

    # Menu: Apply tint
    if action == 'Tint... (T)' or action == 't':
        apply_tint()

    # Menu: Consistency checks
    if action == 'Redundancy and consistency checks... (F9)' or action.startswith('F9:'):
        redundancy_consistency_check()

    # Menu: Scatter coverage plot of palette
    if action == 'Coverage scatter plot...':
        coverage_scatter()

    # Menu: Brightness, contrast, gamma, saturation...
    if action == 'General adjustments... (J)' or action == 'j':
        palette_adjustments()

    # Menu: Auto-gradient
    if action == 'Auto-gradient... (G)' or action == 'g':
        auto_gradient()

    # Menu: Progressive saturation
    if action == 'Progressive saturation...':
        prog_saturation()

    # Menu: Import from paletted image
    if action == 'Import from paletted image...':
        import_paletted_img()

        # Menu: Import from paletted image
    if action == 'Export to paletted image...':
        export_paletted_img()

    # Menu: Merge with another palette
    if action == 'Merge with... (M)' or action == 'm':
        merge_palette()

    # Menu: Sort palette
    if action == 'Sort...':
        sort_palette()

    # Menu: Copy selected color
    if action == 'Copy selected color... (C)' or action == 'c':
        copy_color()

    # Menu: Auto-calculate
    if action == 'Auto-calculate... (F2)' or action.startswith('F2:'):
        auto_calculate_map()

    # Menu: Auto-homogenize
    if action == 'Linear homogenization...':
        lin_homogenize()

    # Menu: Unbalance
    if action == 'Unbalance...':
        unbalance()

    # Menu: Merge with another colormap
    if action == 'Replace map with... (F4)' or action.startswith('F4:'):
        merge_colormap()

    # Menu: Export a colormap to image
    if action == 'Export to image...':
        export_cmp_to_image()

    # Menu: Load colormap from approximated image
    if action == 'Load from image...':
        load_cmp_from_image()

    # Menu: Make selected color luminous
    if action == 'Make selected color luminous (L)' or action == 'l':
        make_sel_luminous()

    # Menu: Mass replace indexes in color map
    if action == 'Mass replace colors... (F8)' or action.startswith('F8:'):
        mass_replace_colormap()

    # Menu: Preview a paletted BMP
    if action == 'Preview bitmap... (F3)' or action.startswith('F3:'):
        preview_pal_bmp()

    # Menu: Solve unevennesses
    if action == 'Solve unevennesses...':
        solve_unevennesses()

    # Menu: Presets
    levpreset = action.rpartition('::')[2]
    if levpreset in Morigs:
        save_undo()
        # Get order number
        levindex = Morigs.index(levpreset)
        Mpalette = copy.deepcopy(cmconst.ORIGS[levindex][2])
        Mcolormap = copy.deepcopy(cmconst.ORIGS[levindex][3])
        draw_all()

    # Menu: Tutorial
    if action == 'Tutorial...':
        webbrowser.open('https://www.youtube.com/watch?v=PYlLuws_JQU')
        continue

    # Menu: Community
    if action == 'Dark Forces community...':
        webbrowser.open('https://www.df-21.net')
        continue

    # Menu: About
    if action == 'About...':
        popup(APPNAME + ' ' + APPVERSION + '\n\n' + cmconst.ABOUTTEXT)
        continue

    # Menu: Whies & becauses
    if action == 'Whys and becauses...':
        popup(cmconst.WHYBECAUSE)
        continue

    # Menu: Keyboard shortcuts
    if action == 'Keyboard shortcuts...':
        popup(cmconst.KEYBOARDSHORTCUTS)
        continue

    # GRAPHS

    # Palette click
    if action == 'palette':
        mx, my = keys['palette']
        # Calculate the new color index accordingly
        newindex = my // 32 * 32 + mx // 32
        if newindex != Mselcolor:
            # Change to new color
            Mselcolor = my // 32 * 32 + mx // 32
            draw_all()
        else:
            # Clicked already selected color - edit it
            modify_color()

    # Map click
    if action == 'colormap':
        mx, my = keys['colormap']
        # Calculate the new color index accordingly
        Mselcolor = mx // 4
        draw_all()

    # Light range clicks
    if action == 'lightrange':
        mx, my = keys['lightrange']
        colindex = mx // 32
        if my <= 15:
            # Color index link; select it
            Mselcolor = Mcolormap[Mselcolor][colindex]
            draw_all()
        else:
            # Adjust specific link in the colormap
            currcolor = Mcolormap[Mselcolor][colindex]
            inftext = 'Enter new color map index for color {0} at lightness {1}:'.format(
                Mselcolor, colindex)
            newcol = sg.popup_get_text(inftext,
                                       title='Remap color',
                                       default_text=str(currcolor),
                                       icon=ICONFILE,
                                       modal=True)
            if not newcol: continue
            # Try to convert to number and get the 'linkage'
            try:
                newcol = round(float(newcol))
                newcol = norm255(newcol)
                save_undo()
                Mcolormap[Mselcolor][colindex] = newcol
                draw_all()
            except BaseException:
                popup('Wrong color index value!')

    # BUTTONS

    # Move selection to previous color
    if action == 'PreviousColor' or action.startswith('Left:'):
        Mselcolor -= 1
        if Mselcolor < 0: Mselcolor = 255
        draw_all()

    # Move selection to next color
    if action == 'NextColor' or action.startswith('Right:'):
        Mselcolor += 1
        if Mselcolor >= 256: Mselcolor = 0
        draw_all()

    # Move selection to previous row
    if action.startswith('Up:'):
        Mselcolor -= 32
        if Mselcolor < 0: Mselcolor += 256
        draw_all()

    # Move selection to next row
    if action.startswith('Down:'):
        Mselcolor += 32
        if Mselcolor >= 256: Mselcolor -= 256
        draw_all()

    # Modify a current color
    if action == 'Modify color' or action == '/':
        modify_color()
