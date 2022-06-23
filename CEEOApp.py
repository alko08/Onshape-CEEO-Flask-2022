from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import io 
import json
import numpy as np 
import os

import base64

import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from onshape_client.client import Client
from onshape_client.onshape_url import OnshapeElement

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

base = 'https://rogers.onshape.com'  # change if using an Enterprise account

# Search and check if a file named "OnshapeAPIKey.py" exists in the folder 
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


@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


# Home page for part assembly extension
@app.route('/home')
def login():
    global EID, WID, DID, STEP, partsDictionary, selected1, selected2, selected3

    STEP = 60
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

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))
    return render_template('home.html', DID=DID, WID=WID, EID=EID, STEP=STEP,  condition1=False,
                           return1=list_parts_assembly(client, url).split('\n'), return2=list(partsDictionary.keys()),
                           return2_len=len(partsDictionary.keys()), selected1=selected1, selected2=selected2,
                           selected3=selected3)


# Graph page for part assembly extension
@app.route('/graph')
def graph():
    global EID, WID, DID, app_key, secret_key, STEP, partsDictionary, selected1, selected2, selected3

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    STEP = float(request.args.get('step'))
    selected1 = request.args.get('rotate_part')
    selected2 = request.args.get('input_track')
    selected3 = request.args.get('output_track')

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})

    input_x_pos = []
    input_y_pos = []
    output_x_pos = []
    output_y_pos = []

    move_id = partsDictionary[selected1]
    in_id = partsDictionary[selected2]
    out_id = partsDictionary[selected3]

    rotation_step = STEP * np.pi / 180  # in radian
    url = '{}/documents/{}/w/{}/e/{}'.format(str(base), str(DID), str(WID), str(EID))

    assembly_info = get_assembly_definition(client, url)
    in_pos = get_position(assembly_info, in_id)
    out_pos = get_position(assembly_info, out_id)
    if in_pos and out_pos:
        input_x_pos.append(in_pos[0])
        input_y_pos.append(in_pos[1])
        output_x_pos.append(out_pos[0])
        output_y_pos.append(out_pos[1])
        for i in range(int(360 / STEP)):
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

    # Plot the path
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(input_x_pos, input_y_pos, label='Input')
    ax.plot(output_x_pos, output_y_pos, label='Output')
    ax.legend()

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return render_template('home.html', image1=base64.b64encode(output.getvalue()).decode("utf-8"), condition1=True,
                           DID=DID, WID=WID, EID=EID, STEP=STEP, return1=list_parts_assembly(client, url).split('\n'),
                           return2=list(partsDictionary.keys()), return2_len=len(partsDictionary.keys()),
                           selected1=selected1, selected2=selected2, selected3=selected3)


# Home page for part workshop extension
@app.route('/home2')
def login2():
    global EID, WID, DID, app_key, secret_key

    view = request.args.get('view_matrix')
    if not view:
        view = "Isometric"

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))
    return render_template('home2.html', DID=DID, WID=WID, EID=EID, img_data=part_studio_shaded_view(client, url, view),
                           condition1=view, return1=list_parts_part_studio(client, url).split('\n'))


@app.route('/home3')
def login3():
    global EID, WID, DID

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')

    frames = 20
    rotation = 360
    zoom_start = (.1001-0.001) * 10000
    zoom_mid = (.1001 - 0.002) * 10000
    zoom_end = (.1001-0.0005) * 10000
    start_view = "Isometric"
    z_auto = True
    loop = True
    zoom3 = False
    no_rotate = False
    direction = 4   # 1=X, 2=Y, 3=XY, 4=Z, 5=XZ, 6=YZ, 7=XYZ
    name = "OnshapeGIF"

    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(base, str(DID), str(WID), str(EID))
    views = get_views(client, url)
    return render_template('home3.html', DID=DID, WID=WID, EID=EID, condition1=False,
                           return1=list_parts_assembly(client, url).split('\n'), FRAMES=frames, ROTATION=rotation,
                           ZSTART=int(zoom_start), ZEND=int(zoom_end), ZAUTO=z_auto, return2=list(views.keys()),
                           return2_len=len(views.keys()), selected1=start_view, LOOP=loop, ZOOM3=zoom3, NAME=name,
                           ZMID=int(zoom_mid), DIRECTION=int(direction), ROTATE=no_rotate)


# Graph page for part assembly extension
@app.route('/gif')
def gif():
    global EID, WID, DID, app_key, secret_key, partsDictionary

    did = request.args.get('documentId')
    wid = request.args.get('workspaceId')
    eid = request.args.get('elementId')
    frames = int(request.args.get('frames'))
    rotation = float(request.args.get('rotation'))
    z_auto = bool(request.args.get('zoom_auto'))
    zoom_start = .1001 - float(request.args.get('zoom_start')) / 10000
    zoom_end = .1001 - float(request.args.get('zoom_end')) / 10000
    zoom_mid = .1001 - float(request.args.get('zoom_mid')) / 10000
    start_view = request.args.get('start_view')
    loop = bool(request.args.get('loop'))
    zoom3 = bool(request.args.get('do_zoom_mid'))
    direction = 0 + bool(request.args.get('rotateX'))
    direction = direction + 2 * bool(request.args.get('rotateY'))
    direction = direction + 4 * bool(request.args.get('rotateZ'))
    name = request.args.get('name')

    if did or wid or eid:
        DID = did
        WID = wid
        EID = eid

    client = Client(configuration={"base_url": base, "access_key": app_key, "secret_key": secret_key})
    url = '{}/documents/{}/w/{}/e/{}'.format(str(base), str(DID), str(WID), str(EID))

    views = get_views(client, url)
    return render_template('home3.html', condition1=True, DID=DID, WID=WID, EID=EID, FRAMES=frames, ROTATION=rotation,
                           image1=stepping_rotation(client, url, frames, rotation, zoom_start, zoom_end, start_view,
                                                    z_auto, loop, zoom3, zoom_mid, direction, name),
                           return1=list_parts_assembly(client, url).split('\n'), ZSTART=int((.1001-zoom_start)*10000),
                           ZEND=int((.1001-zoom_end)*10000), return2=list(views.keys()), return2_len=len(views.keys()),
                           selected1=start_view, ZAUTO=z_auto, LOOP=loop, ZOOM3=zoom3, NAME=name,
                           ZMID=int((.1001-zoom_mid)*10000), DIRECTION=int(direction))


# -------------------------------------------------------------------------------------------#
# ------------------ Helper Functions -------------------------------------------------------#
# -------------------------------------------------------------------------------------------#
def rotate_input(client, assembly, url: str, part_id: str, rotation: float):
    """
    This function rotates the input link of the mechanism 
    with a fixed rotation step in degree; changes are 
    made to the actual model 
    """
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


def get_assembly_definition(client, url: str):
    """
    This function gets the definition of the assembly, 
    including information of all part instances and mate features 
    """
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


def get_position(assembly, part_id: str):
    """
    This function parses through all the parts within the assembly 
    and returns the x and y positions of the position trackers specified 
    with the partId. 
    """
    for occ in assembly['rootAssembly']['occurrences']: 
        if occ['path'][0] == part_id:
            return occ['transform'][3], occ['transform'][7]
    print("Part not found!") 
    return None 


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


def list_parts_part_studio(client, url):
    output_html = ""
    part_response = get_parts_in_document(client, url)

    for i in range(len(part_response)):
        output_html = output_html + "" + (part_response[i]["name"] + "\nPart ID: " + part_response[i]["partId"] +
                                          "\nElement ID: " + part_response[i]["elementId"] + "\n\n")

    return output_html


# Get Shaded View of PartStudio, returns the base64 image string of a shaded view of a part studio
def part_studio_shaded_view(client, url: str, view_matrix="front"):
    # viewMatrix can be any face direction or isometric as a string, or a 1x12 view matrix, type:"string"
    # pixelSize is the size in meters for each pixel. If 0, it will fill the image size output, type:"number"

    fixed_url = '/api/partstudios/d/did/w/wid/e/eid/shadedviews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'

    matrix = "0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0"
    if any(face in view_matrix for face in ["Front", "Back", "Top", "Bottom", "Left", "Right"]):
        matrix = view_matrix
    elif view_matrix == "Flipped Isometric":
        matrix = "0.612,0.612,0,0,0.354,-0.354,-0.707,0,-0.707,0.707,-0.707,0"
    elif isinstance(view_matrix, list):
        matrix = str(matrix).replace('[', '').replace(']', '')

    # View Matrix below is roughly isometric
    params = {'viewMatrix': matrix,
              'edges': 'show',
              'outputHeight': 600,
              'outputWidth': 1000,
              'pixelSize': 0.001}
    # print(params)
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

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
def assemblies_shaded_view(client, url: str, view_matrix="Isometric", pixel_size=0.000, edges="show",
                           filename="image.jpg", output_height=600, output_width=1000):
    fixed_url = '/api/assemblies/d/did/w/wid/e/eid/shadedviews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('wid', element.wvmid)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'
    matrix = "0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0"
    if any(face in view_matrix for face in ["front", "back", "top", "bottom", "left", "right"]):
        matrix = view_matrix
    elif isinstance(view_matrix, list):
        matrix = str(view_matrix).replace('[', '').replace(']', '')

    # View Matrix below is roughly isometric
    params = {'viewMatrix': matrix,
              'edges': edges,
              'outputHeight': output_height,
              'outputWidth': output_width,
              'pixelSize': pixel_size}
    # print(params)
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    parsed = json.loads(response.data)
    img = base64.b64decode(parsed['images'][0])
    with open(filename, 'wb') as f:
        f.write(img)

    return img


# -------------------------------------#
# ----View Matrix Helper Functions-----#
# -------------------------------------#
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


# identity_twelve() returns a flattened identity view matrix (1x12)
def identity_twelve():
    m = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
        ]
    return m


# identity_twelve() returns a flattened identity view matrix (1x12)
def identity_fourxthree():
    m = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ]
    return m


# move_matrix(base,x1,y1,z1) takes a 1x12 view matrix and moves the x,y,z coordinates
def move_matrix(matrix_base, x1, y1, z1):
    matrix = matrix_base
    matrix[0][3] = x1
    matrix[1][3] = y1
    matrix[2][3] = z1
    return matrix


# move_flat(base,x1,y1,z1) takes a 1x12 view matrix and moves the x,y,z coordinates
def move_flat(matrix_base, x1, y1, z1):
    matrix = matrix_base
    matrix[3] = x1
    matrix[7] = y1
    matrix[11] = z1
    return matrix


# twelve_threexfour(matrix) takes a flattened 1x12 view matrix and makes a 4x3 matrix for linear algebra
def twelve_fourxthree(matrix):
    threexfour = [[matrix[0], matrix[1], matrix[2], matrix[3]],
                  [matrix[4], matrix[5], matrix[6], matrix[7]],
                  [matrix[8], matrix[9], matrix[10], matrix[11]]]
    return threexfour


# threexfour_twelve(matrix) takes a 4x3 view matrix and flattens it to 1x12, the form used by Onshape
def fourxthree_twelve(matrix):
    twelve = [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
              matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
              matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]]
    return twelve


def fourxfour_fourxthree(matrix):
    matrix.pop(3)
    return matrix


# clockwise_spin(theta) returns a 4x3 matrix with a rotation of theta around the specified direction.
# 1=X, 2=Y, 3=XY, 4=Z, 5=XZ, 6=YZ, 7=XYZ
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
    fixed_url = '/api/assemblies/d/did/e/eid/namedViews'
    element = OnshapeElement(url)
    fixed_url = fixed_url.replace('did', element.did)
    fixed_url = fixed_url.replace('eid', element.eid)

    method = 'GET'

    # View Matrix below is roughly isometric
    params = {}
    # print(params)
    payload = {}
    headers = {'Accept': 'application/vnd.onshape.v1+json',
               'Content-Type': 'application/json'}

    response = client.api_client.request(method, url=base + fixed_url, query_params=params, headers=headers,
                                         body=payload)

    parsed = json.loads(response.data)

    # views = assembliesNamedViews(url)
    # print(json.dumps(views, indent=4, sort_keys=True))

    return parsed


# get_views(client, url: str) returns list of all named and regular views
def get_views(client, url: str):
    global viewsDictionary
    view_matrices = assemblies_named_views(client, url)['namedViews']

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
# ------View Matrix Functions----------#
# -------------------------------------#
def stepping_rotation(client, url: str, frames=60, rotation=45.0, zoom_start=0.001, zoom_end=0.0005,
                      start_view="Isometric", z_auto=False, loop=True, zoom3=False, zoom_mid=0.002, direction=2,
                      name="OnshapeGIF"):
    global viewsDictionary

    if direction >= 7:
        rotation = rotation / np.sqrt(3)
    elif direction >= 3 and direction != 4:
        rotation = rotation / np.sqrt(2)

    if rotation == 0:
        total_z_rotation_angle = 0
    else:
        total_z_rotation_angle = np.pi / (180 / rotation)
    translation_start = [0, 0, 0]
    translation_end = [0, 0, -0.05]

    view_array = viewsDictionary[start_view]

    # Build new array from old array
    new_array = [view_array[0:4], view_array[4:8], view_array[8:12]]

    images = []
    matrix = new_array
    zoom_array2 = []

    translation = np.linspace(translation_start, translation_end, frames)

    if not z_auto:
        if zoom3:
            zoom_array = np.linspace(zoom_start, zoom_mid, int(frames/2+.5))
            zoom_array2 = np.linspace(zoom_mid, zoom_end, int(frames/2))
        else:
            zoom_array = np.linspace(zoom_start, zoom_end, frames)
    else:
        zoom_array = np.linspace(0, 0, frames)

    matrix = multiply(matrix, clockwise_spin(total_z_rotation_angle / frames, direction))
    matrix = move_matrix(matrix, translation[0][0], translation[0][1], translation[0][2])
    flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()
    assemblies_shaded_view(client, url, flattened, zoom_array[0], "hide", "image.jpg")
    im1 = gen_frame("image.jpg")
    for i in range(1, frames):
        matrix = multiply(matrix, clockwise_spin(total_z_rotation_angle / frames, direction))
        matrix = move_matrix(matrix, translation[i][0], translation[i][1], translation[i][2])
        flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()
        if zoom3 and i >= len(zoom_array):
            # print(str(i) + " | " + str(len(zoom_array2)))
            assemblies_shaded_view(client, url, flattened, zoom_array2[i - len(zoom_array)], "hide", "image.jpg")
        else:
            # print(str(i) + " | " + str(len(zoom_array)))
            assemblies_shaded_view(client, url, flattened, zoom_array[i], "hide", "image.jpg")
        images.append(gen_frame("image.jpg"))
        print(str(int(i/frames * 1000)/10) + "%", end="\r")
    print("")
    if loop:
        im1.save('static/images/'+name+'.gif', save_all=True, loop=0, append_images=images, disposal=2, duration=0)
    else:
        im1.save('static/images/'+name+'.gif', save_all=True, append_images=images, disposal=2, duration=0)
    return 'static/images/'+name+'.gif'


# def linear_interpolation(client, url: str):
#     start = 'view 1'
#     end = 'view 4'
#
#     view_matrices = assemblies_named_views(client, url)
#     view1 = view_matrices['namedViews'][start]['viewMatrix']
#     view2 = view_matrices['namedViews'][end]['viewMatrix']
#
#     # Build new array from old array
#     new1 = [view1[0:4], view1[4:8], view1[8:12]]
#     new2 = [view2[0:4], view2[4:8], view2[8:12]]
#
#     array = np.linspace(new1, new2, 25)
#
#     images = []
#     matrix = array[0]
#     flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()
#     assemblies_shaded_view(client, url, flattened, 0.001, "hide", "image.jpg")
#     im1 = gen_frame("image.jpg")
#     for matrix in array[1:]:
#         flattened = matrix[0][0:4].tolist() + matrix[1][0:4].tolist() + matrix[2][0:4].tolist()
#         assemblies_shaded_view(client, url, flattened, 0.001, "hide", "image.jpg")
#         images.append(gen_frame("image.jpg"))
#
#     im1.save('static/images/OnshapeGIF2.gif', save_all=True, loop=0, append_images=images, disposal=2, duration=0)
#     return 'static/images/OnshapeGIF2.gif'
