# Imports needed for multiple extensions and file structure as a whole
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import json
import numpy as np
import os
import base64

# Imports for Onshape API Calls
from onshape_client.client import Client
from onshape_client.onshape_url import OnshapeElement

# Imports for the CEEO Rotate and Graph Extension
import io
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Imports for the CEEO GIF Maker Extension
from PIL import Image

# -------------------------------------------------------------------------------------------#
# ------------------ Defining Variables -----------------------------------------------------#
# -------------------------------------------------------------------------------------------#
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['figure.autolayout'] = True 
app = Flask(__name__)

app_key = ''
secret_key = ''
DID = ''
WID = '' 
EID = ''
STEP = 60
partsDictionary = {}
viewsDictionary = {}
selected1 = "Input <1>"
selected2 = "Position Tracker <1>"
selected3 = "Position Tracker <2>"

base = 'https://rogers.onshape.com'  # Change if using an Enterprise account
# base = 'https://cad.onshape.com'  # This is the default Enterprise

# Search and check if a file named "OnshapeAPIKey.py" exists in the folder. Then uses the API Keys found in the file
for _, _, files in os.walk('.'): 
    if "OnshapeAPIKey.py" in files: 
        exec(open('OnshapeAPIKey.py').read())
        app_key = access
        secret_key = secret
        break 


# -------------------------------------------------------------------------------------------#
# ------------------ URL Functions ----------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
@app.route('/')
def index():
    return redirect(url_for("home"))


# This allows and the user to access all files in static, like images!
@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


# Home page for part assembly, CEEO Rotate extension
@app.route('/home')
def login():
    global EID, WID, DID, STEP, partsDictionary, selected1, selected2, selected3

    # Defines default values
    STEP = 6
    selected1 = "Input <1>"
    selected2 = "Position Tracker <1>"
    selected3 = "Position Tracker <2>"

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    # Generate Onshape URL and client for API calls
    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))

    # Returns html webpage and make api calls using template 'home.html'
    return render_template('home.html', DID=DID, WID=WID, EID=EID, STEP=STEP, condition1=False,
                           return1=list_parts_assembly(client, url).split('\n'), return2=list(partsDictionary.keys()),
                           return2_len=len(partsDictionary.keys()), selected1=selected1, selected2=selected2,
                           selected3=selected3)


# Graph page for part assembly, CEEO Rotate extension. Almost the exact same as home, but takes in input values and
# sends them to the graph function to make and return a graph
@app.route('/graph')
def graph():
    global EID, WID, DID, app_key, secret_key, STEP, partsDictionary, selected1, selected2, selected3

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    # Only redefine global Onshape ID's if the request ID's returns a new ID
    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    # Receive inputs from User
    STEP = int(request.args.get('step'))   # Step value for rotation amount
    selected1 = request.args.get('rotate_part')   # What part to rotate
    selected2 = request.args.get('input_track')   # What part to track and graph as input
    selected3 = request.args.get('output_track')   # What part to track and graph as output

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})

    # ------ START OF ROTATION CODE ------ #
    # Define variables
    input_x_pos = []
    input_y_pos = []
    output_x_pos = []
    output_y_pos = []

    move_id = partsDictionary[selected1]
    in_id = partsDictionary[selected2]
    out_id = partsDictionary[selected3]

    # Creating rotation step
    rotation_step = 2 * np.pi / STEP  # in radian
    url = '{}/documents/{}/w/{}/e/{}'.format(str(base), str(DID), str(WID), str(EID))

    # Check for initial positions and assembly info
    assembly_info = get_assembly_definition(client, url)
    in_pos = get_position(assembly_info, in_id)
    out_pos = get_position(assembly_info, out_id)
    if in_pos and out_pos:
        # Add initial positions to position array
        input_x_pos.append(in_pos[0])
        input_y_pos.append(in_pos[1])
        output_x_pos.append(out_pos[0])
        output_y_pos.append(out_pos[1])
        for i in range(STEP):
            # Rotate the input by rotation_step
            rotate_input(client, assembly_info, url, move_id, rotation_step)
            # Get the x-y position of the input and output position trackers
            assembly_info = get_assembly_definition(client, url)
            in_pos = get_position(assembly_info, in_id)
            out_pos = get_position(assembly_info, out_id)
            input_x_pos.append(in_pos[0])
            input_y_pos.append(in_pos[1])
            output_x_pos.append(out_pos[0])
            output_y_pos.append(out_pos[1])

    # Plot the path of the input and output positions data
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(input_x_pos, input_y_pos, label='Input')
    ax.plot(output_x_pos, output_y_pos, label='Output')
    ax.legend()

    # Send output image to user using template 'home.html'
    # Also makes sure to set all variables to what they submitted
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return render_template('home.html', image1=base64.b64encode(output.getvalue()).decode("utf-8"), condition1=True,
                           DID=DID, WID=WID, EID=EID, STEP=STEP, return1=list_parts_assembly(client, url).split('\n'),
                           return2=list(partsDictionary.keys()), return2_len=len(partsDictionary.keys()),
                           selected1=selected1, selected2=selected2, selected3=selected3)


# Home page for part studio extension
@app.route('/home2')
def login2():
    global EID, WID, DID, app_key, secret_key

    # Request user input (view). If nothing is returned set view to Isometric
    view = request.args.get('view_matrix')
    if not view:
        view = "Isometric"

    # Get Onshape ID's
    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    # Only redefine global Onshape ID's if the request ID's returns a new ID
    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    # Send output image to user using template 'home2.html'
    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))
    return render_template('home2.html', DID=DID, WID=WID, EID=EID, img_data=part_studio_shaded_view(client, url, view),
                           condition1=view, return1=list_parts_part_studio(client, url).split('\n'))


# Home page for CEEO GIF Maker, assembly extension.
@app.route('/home3')
def login3():
    global EID, WID, DID

    # Request Onshape IDs
    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    # Only redefine global Onshape ID's if the request ID's returns a new ID
    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    # Set default input values
    frames = 20   # Frames in GIF
    rotation = 360   # Degrees GIF rotates
    # Behind the scenes the values of zooms are like 0.001, but then converted to make it easier to read for the user
    zoom_start = (.1001-0.001) * 10000
    zoom_mid = (.1001 - 0.002) * 10000
    zoom_end = (.1001-0.0005) * 10000
    start_view = "Isometric"
    z_auto = False   # Auto Zoom
    loop = True   # GIF Loops
    zoom3 = False   # Midpoint zoom
    zoom2 = False   # End Zoom
    direction = 1   # 1=Z, 2=Y, 3=ZY, 4=X, 5=XZ, 6=YX, 7=XYZ as x & y swapped for this app.
    name = "OnshapeGIF"   # Filename
    duration = 0   # Duration of each frame
    edges = False   # Show edges
    height = 600   # Height of GIF, pixels
    width = 1000   # Width of GIF, pixels

    # Send output to user using template 'home3.html'. Output does not include GIF,
    # but does include default input values and list of parts through list_parts_assembly()
    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))
    views = get_views(client, url)
    return render_template('home3.html', DID=DID, WID=WID, EID=EID, condition1=False, EDGES=edges, HEIGHT=height,
                           return1=list_parts_assembly(client, url).split('\n'), FRAMES=frames, ROTATION=rotation,
                           ZSTART=int(zoom_start), ZEND=int(zoom_end), ZAUTO=z_auto, return2=list(views.keys()),
                           return2_len=len(views.keys()), selected1=start_view, LOOP=loop, ZOOM3=zoom3, NAME=name,
                           ZMID=int(zoom_mid), DIRECTION=int(direction), DURATION=duration, ZOOM2=zoom2, WIDTH=width)


# GIF page for CEEO GIF Maker, assembly extension. Called after user inputs values to return a GIF.
@app.route('/gif')
def gif():
    global EID, WID, DID, app_key, secret_key, partsDictionary

    # Request Onshape IDs
    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    # Only redefine global Onshape ID's if the request ID's returns a new ID
    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    # ------ Request and format user inputs ------ #
    frames = int(request.args.get('frames'))
    loop = bool(request.args.get('loop'))
    name = request.args.get('name')
    duration = int(request.args.get('duration'))

    # For the rest of the code X is 1, Y is 2, and Z is 4, but for some reason when the GIF Maker runs it swaps the X
    # and Z axis. Either this is some result of the shaded view API call or something with the code. To fix this and
    # not break the rotate functions, the X and Z value are swapped here
    rotation = float(request.args.get('rotation'))
    direction = 0 + 4 * bool(request.args.get('rotateX'))
    direction = direction + 2 * bool(request.args.get('rotateY'))
    direction = direction + 1 * bool(request.args.get('rotateZ'))

    z_auto = bool(request.args.get('zoom_auto'))
    zoom2 = bool(request.args.get('do_zoom_end'))
    zoom3 = bool(request.args.get('do_zoom_mid'))
    start_view = request.args.get('start_view')

    # Convert zoom values back to expected values of decimals, also determines if zoom start, end, and mid is needed.
    if z_auto:
        zoom_start = 0   # Auto zoom when zoom = 0
        zoom2 = False
    else:
        zoom_start = .1001 - float(request.args.get('zoom_start')) / 10000

    if not zoom2:
        zoom_end = zoom_start
        zoom_mid = zoom_start
    else:
        zoom_end = .1001 - float(request.args.get('zoom_end')) / 10000
        zoom_mid = .1001 - float(request.args.get('zoom_mid')) / 10000

    edges = bool(request.args.get('edges'))
    height = int(request.args.get('height'))
    width = int(request.args.get('width'))
    # ------ End of user inputs ------ #

    # Generating Onshape API information
    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(str(base), str(DID), str(WID), str(EID))

    # Send output to user using template 'home3.html'. Output includes created GIF, and won't send until its done.
    # Also makes sure to set all variables to what they submitted, includes converting back for zooms
    views = get_views(client, url)
    return render_template('home3.html', condition1=True, DID=DID, WID=WID, EID=EID, FRAMES=frames, ROTATION=rotation,
                           image1=stepping_rotation(client, url, frames, rotation, zoom_start, zoom_end, start_view,
                                                    loop, zoom3, zoom_mid, direction, name, duration, edges,
                                                    height, width),
                           return1=list_parts_assembly(client, url).split('\n'), ZSTART=int((.1001-zoom_start)*10000),
                           ZEND=int((.1001-zoom_end)*10000), return2=list(views.keys()), return2_len=len(views.keys()),
                           selected1=start_view, ZAUTO=z_auto, LOOP=loop, ZOOM3=zoom3, NAME=name, DURATION=duration,
                           ZMID=int((.1001-zoom_mid)*10000), DIRECTION=int(direction), ZOOM2=zoom2, EDGES=edges,
                           HEIGHT=height, WIDTH=width)


# -------------------------------------------------------------------------------------------#
# ------------------ Helper Functions -------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
# This function rotates the input link of the mechanism with a fixed rotation step in degree; changes are
# made to the actual model, credit to: Felix Deng @ https://github.com/PTC-Education/Four-Bar-Mechanism
def rotate_input(client, assembly, url: str, part_id: str, rotation: float):

    identity_matrix = np.reshape([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], (4, 4))
    occurrences = assembly['rootAssembly']['occurrences']
    occurrence = None
    for x in occurrences:
        if x['path'][0] == part_id:
            occurrence = x
    if not occurrence: 
        print("Part not found!")
        return None
    
    rot_mat = np.matmul(identity_matrix, clockwise_spinz(rotation))
    transform_mat = np.matmul(identity_matrix, rot_mat)

    fixed_url = '/api/assemblies/d/did/w/wid/e/eid/occurrencetransforms'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'POST'
    params = {}
    payload = {'isRelative': True,
               'occurrences': [occurrence],
               'transform': list(np.reshape(transform_mat, -1))}
    headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1',
               'Content-Type': 'application/json'}

    client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers, body=payload)


# This function gets the definition of the assembly, including information of all part instances and mate features
def get_assembly_definition(client, url: str):
    fixed_url = '/api/assemblies/d/did/w/wid/e/eid'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'
    params = {}
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1', 'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)
    parsed = json.loads(response.data)
    return parsed


# This function parses through all the parts within the assembly and returns the x and y positions of the position
# trackers specified with the partId.
def get_position(assembly, part_id: str):
    for occ in assembly['rootAssembly']['occurrences']: 
        if occ['path'][0] == part_id:
            return occ['transform'][3], occ['transform'][7]
    print("Part not found!") 
    return None 


# This functions requests the assembly definition
# Creates a dictionary of all parts by "name" = "ID" to global partsDictionary
# Returns the parts as output html that can be listed for the user.
def list_parts_assembly(client, url):
    global partsDictionary
    output_html = ""
    part_response = get_assembly_definition(client, url)

    partsDictionary.clear()
    for instance in part_response['rootAssembly']['instances']:
        output_html = output_html + "" + (
                    instance["name"] + "\nID: " + instance["id"] + "\n\n")
        partsDictionary[instance["name"]] = instance["id"]
    return output_html


# This functions requests the part studio json list of parts through get_parts_in_document() and then converts it into
# output html that can be listed for the user.
def list_parts_part_studio(client, url):
    output_html = ""
    part_response = get_parts_in_document(client, url)

    for i in range(len(part_response)):
        output_html = output_html + "" + (part_response[i]["name"] + "\nPart ID: " + part_response[i]["partId"] +
                                          "\nElement ID: " + part_response[i]["elementId"] + "\n\n")

    return output_html


# Get Shaded View of PartStudio, returns the base64 image string of a shaded view of a part studio
# viewMatrix can be any face direction or isometric as a string, or a 1x12 view matrix, type:"string"
def part_studio_shaded_view(client, url: str, view_matrix="front"):
    # fixed url used for API request
    fixed_url = '/api/partstudios/d/did/w/wid/e/eid/shadedviews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'

    # Basic Isometric Matrix
    matrix = "0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0"
    if any(face in view_matrix for face in ["Front", "Back", "Top", "Bottom", "Left", "Right"]):
        matrix = view_matrix   # Onshape client will accept one of these six strings as just a word instead of an array
    elif view_matrix == "Flipped_Isometric":   # Custom Matrix created to be a flipped version of the isometric matrix
        matrix = "0.612,0.612,0,0,0.354,-0.354,-0.707,0,-0.707,0.707,-0.707,0"
    elif isinstance(view_matrix, list):   # Else if given a list, convert it into a string.
        matrix = str(view_matrix).replace('[', '').replace(']', '')

    params = {'viewMatrix': matrix,
              'edges': 'show',
              'outputHeight': 600,
              'outputWidth': 1000,
              'pixelSize': 0}   # Pixel Size = 0, so it automatically zooms

    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    # Load the image from the Onshape API request, encode and decode it (using base64) to send to user
    parsed = json.loads(response.data)
    img_data = base64.b64decode(parsed['images'][0])
    img_data = base64.b64encode(img_data).decode("utf-8")
    return img_data


# Get Parts in Document, returns JSON of all parts in a part studio
def get_parts_in_document(client, url: str):
    fixed_url = '/api/parts/d/did/w/wid/e/eid/'

    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'

    params = {}
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1', 'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    parsed = json.loads(response.data)
    # The command below prints the entire JSON response from Onshape
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    return parsed


# -----------------------------------------------------#
# ------------ Assembly Gif Functions -----------------#
# -----------------------------------------------------#
# This function returns a shaded view of the provided assembly with the provided settings. These settings are extremely
# customizable. It allows you to change the view angle, pixels size (zoom), if edges are "show" or "hide", the filename
# of the saved image, the image height and width in pixels.
def assemblies_shaded_view(client, url: str, view_matrix="Isometric", pixel_size=0.000, edges="show",
                           filename="image.jpg", output_height=600, output_width=1000):
    # Fixed URL for Onshape assemblies, shaded view API.
    fixed_url = '/api/assemblies/d/did/w/wid/e/eid/shadedviews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'
    matrix = "0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0"   # default Isometric View
    if any(face in view_matrix for face in ["front", "back", "top", "bottom", "left", "right"]):
        matrix = view_matrix   # Onshape client will accept one of these six strings as just a word instead of an array
    elif isinstance(view_matrix, list):   # Else if given a list, convert it into a string.
        matrix = str(view_matrix).replace('[', '').replace(']', '')

    # Create params based on all given inputs.
    params = {'viewMatrix': matrix,
              'edges': edges,
              'outputHeight': output_height,
              'outputWidth': output_width,
              'pixelSize': pixel_size}

    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    # Load image based on response data and encode plus decode it using base64
    parsed = json.loads(response.data)
    img = base64.b64decode(parsed['images'][0])
    with open(filename, 'wb') as f:
        f.write(img)

    return img


# -------------------------------------#
# ----View Matrix Helper Functions-----#
# -------------------------------------#
# Transforms image from file path into a frame that can be used in the GIF maker
def gen_frame(path):
    im = Image.open(path)
    alpha = im.getchannel('A')

    # Convert the image into P mode but only use 255 colors in the palette out of 256
    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

    # Set all pixel values below 5 to 255 , and the rest to 0
    mask = Image.eval(alpha, lambda a: 255 if a <= 5 else 0)

    # Paste the color of index 255 and use alpha as a mask
    im.paste(255, mask)

    # The transparency index is 255
    im.info['transparency'] = 255

    return im


# multiply(x,y) multiplies two 4x3 view matrices to get their determinant
def multiply(x, y):
    result = np.matmul(x, y)
    return result


# identity_fourxthree() returns an identity view matrix (4x3)
def identity_fourxthree():
    m = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ]
    return m


# clockwise_spin(theta) returns a 4x3 matrix with a rotation of theta around the specified direction.
# 1=X, 2=Y, 3=XY, 4=Z, 5=XZ, 6=YZ, 7=XYZ. Does this by starting at biggest value and subtracting each time
def clockwise_spin(theta, direction):
    m = identity_fourxthree()
    if direction >= 4:
        m = clockwise_spinx(theta)
        direction -= 4
    if direction >= 2:
        m = multiply(m, clockwise_spiny(theta))
        direction -= 2
    if direction >= 1:
        m = multiply(m, clockwise_spinz(theta))
    return m


# clockwise_spinx(theta) returns a 4x3 matrix with a rotation of theta around the x axis.
def clockwise_spinx(theta):
    m = [[1, 0, 0, 0],
         [0, np.cos(theta), np.sin(theta), 0],
         [0, -np.sin(theta), np.cos(theta), 0],
         [0, 0, 0, 1]
         ]
    return m


# clockwise_spiny(theta) returns a 4x3 matrix with a rotation of theta around the y axis.
def clockwise_spiny(theta):
    m = [[np.cos(theta), 0, np.sin(theta), 0],
         [0, 1, 0, 0],
         [-np.sin(theta), 0, np.cos(theta), 0],
         [0, 0, 0, 1]
         ]
    return m


# clockwise_spinz(theta) returns a 4x3 matrix with a rotation of theta around the z axis.
def clockwise_spinz(theta):
    m = [[np.cos(theta), np.sin(theta), 0, 0],
         [-np.sin(theta), np.cos(theta), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    return m


# Get named views from Assembly
# assemblies_named_views(client, url: str) returns JSON of all named views in an assembly
def assemblies_named_views(client, url: str):
    # Fixed URL for Onshape assemblies, named views API.
    fixed_url = '/api/assemblies/d/did/e/eid/namedViews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'

    params = {}
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    # make Onshape API call
    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    # load json response and return it
    parsed = json.loads(response.data)
    return parsed


# get_views(client, url: str) returns list of all named and regular views
# The regular views are hard coded in. The data was gotten through the use of named views in the exact same positions
def get_views(client, url: str):
    global viewsDictionary
    view_matrices = assemblies_named_views(client, url)['namedViews']

    # clears the global dictionary (viewsDictionary) and then refills its values
    # Named views first, then hard coded default views
    viewsDictionary.clear()
    if view_matrices:
        for a in view_matrices:
            viewsDictionary[a] = view_matrices[a]['viewMatrix']
    viewsDictionary["Front"] = [1.0, 0.0, -0.0, 0.0, 0.0, 2.220446049250313e-16, 1.0, 0.0, 0.0, -1.0,
                                2.220446049250313e-16, 0.0, 0.1749407045420097, -0.3300952081909283,
                                0.015000000000000013, 1.0]
    viewsDictionary["Back"] = [-1.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313e-16, 1.0, 0.0, 0.0, 1.0,
                               -2.220446049250313e-16, 0.0, 0.1749407045420097, 0.7234216773965487,
                               0.015000000000000124, 1.0]
    viewsDictionary["Top"] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.17494070454200972,
                              0.19666323460281035, 0.5417584427937385, 1.0]
    viewsDictionary["Bottom"] = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.17494070454200963,
                                 0.19666323460281018, -0.5117584427937385, 1.0]
    viewsDictionary["Left"] = [1.9721522630525278e-31, -1.0, -4.440892098500624e-16, 0.0, 0.0, -4.440892098500624e-16,
                               1.0, 0.0, -1.0, 0.0, -4.440892098500624e-16, 0.0, -0.3518177382517288,
                               0.19666323460281032, 0.01500000000000002, 1.0]
    viewsDictionary["Right"] = [1.9721522630525278e-31, -1.0, -4.440892098500624e-16, 0.0, 0.0, -4.440892098500624e-16,
                                1.0, 0.0, -1.0, 0.0, -4.440892098500624e-16, 0.0, -0.3518177382517288,
                                0.19666323460281032, 0.01500000000000002, 1.0]
    viewsDictionary["Isometric"] = [0.7071067811865475, 0.7071067811865478, 1.6653345369377348e-16, 0.0,
                                    -0.40824829046386313, 0.4082482904638628, 0.8164965809277261, 0.0,
                                    0.577350269189626, -0.5773502691896258, 0.5773502691896257, 0.0,
                                    0.4790648332868827, -0.10746089414206281, 0.3191241287448731, 1.0]
    return viewsDictionary


# -------------------------------------#
# ------Gif Creating Function----------#
# -------------------------------------#
"""
# This function creates a GIF. Its main purpose is to create a GIF where the camera rotates around the object to give
# the object the effect that it is spinning. It also has the options to zoom in and out throughout the GIF by setting
# the zoom start/end, along with an option to set the middle zoom point if zoom3 is set to True. This means this
# function is multipurpose, set direction to 0 to make it not rotate and only zoom in and out. Set zooms to 0 to have,
# it automatically zoom. Set frames to 1 to just get a image saved as jpg. Change if edges are shown, what the height
# and width of the GIF will be, along with the name of the file. Also change how long each frame is shown with duration.
"""


# Integers: frames, direction, duration, height, width
# Floats: rotation, zoom_start, zoom_end, zoom_mid
# Strings: start_view must be a  name from viewsDictionary, name
# Booleans: loop, edges, zoom3
# Special: "client" must be an Onshape client and "url" the url of an Onshape Assembly
def stepping_rotation(client, url: str, frames=60, rotation=360.0, zoom_start=0.001, zoom_end=0.001,
                      start_view="Isometric", loop=True, zoom3=False, zoom_mid=0.002, direction=4,
                      name="OnshapeGIF", duration=0, edges=False, height=600, width=1000):
    global viewsDictionary

    if direction >= 7:
        # If direction is all three directions, due to geometry need to decide rotation by root 3
        rotation = rotation / np.sqrt(3)
    elif direction >= 3 and direction != 4:
        # If direction is in two directions, due to geometry need to decide rotation by root 2
        rotation = rotation / np.sqrt(2)
    elif direction == 0:
        # If direction is 0, set rotation to 0
        rotation = 0

    # Simple check to prevent dividing by 0
    if rotation == 0:
        total_z_rotation_angle = 0
    else:
        total_z_rotation_angle = np.pi / (180 / rotation)

    # Convert edges from boolean to string of "show" and "hide" for Onshape API
    if edges:
        edges = "show"
    else:
        edges = "hide"

    # Create the view array by finding the corresponding array using the view dictionary.
    view_array = viewsDictionary[start_view]

    # Build new array from old array
    new_array = [view_array[0:4], view_array[4:8], view_array[8:12]]

    # Defining variables
    images = []
    matrix = new_array

    # Creates zoom array. Checks if midpoint zoom exists or not.
    if zoom3:
        zoom_array = np.linspace(zoom_start, zoom_mid, int(frames / 2 + .5))
        zoom_array2 = np.linspace(zoom_mid, zoom_end, int(frames / 2))
        zoom_array = np.append(zoom_array, zoom_array2)
    else:
        zoom_array = np.linspace(zoom_start, zoom_end, frames)

    # First frame is generated. Slight different then creating frames inside the for loop
    matrix = multiply(matrix, clockwise_spin(total_z_rotation_angle / frames, direction))   # Spin matrix
    flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()   # Flatten matrix
    assemblies_shaded_view(client, url, flattened, zoom_array[0], edges, "image.jpg", height, width)   # Get frame
    im1 = gen_frame("image.jpg")    # This is the only difference, save first frame as base image
    print(str(int(1 / frames * 1000) / 10) + "%", end="\r")   # print progress in format "100.0%"

    # For loop to iterate through the rest of the frames.
    for i in range(1, frames):
        matrix = multiply(matrix, clockwise_spin(total_z_rotation_angle / frames, direction))   # Spin matrix
        flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()   # Flatten matrix
        assemblies_shaded_view(client, url, flattened, zoom_array[i], edges, "image.jpg", height, width)   # Get frame
        images.append(gen_frame("image.jpg"))   # Save frame to images list
        print(str(int((i + 1)/frames * 1000)/10) + "%", end="\r")   # print progress in format "100.0%"

    print("")   # prints an empty line as progress is 100.0%
    if frames == 1:   # If only one frame, send the user a jpg, not a gif
        os.rename("image.jpg", 'static/images/' + name + '.jpg')
        return 'static/images/' + name + '.jpg'

    if loop:   # If loop is set to true, loop gif multiple times, else don't loop
        im1.save('static/images/'+name+'.gif', save_all=True, loop=0, append_images=images, disposal=2,
                 duration=duration)
    else:
        im1.save('static/images/'+name+'.gif', save_all=True, append_images=images, disposal=2, duration=duration)
    return 'static/images/'+name+'.gif'   # Return GIF
