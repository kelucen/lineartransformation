import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import os
import base64
import cv2

from flask import Flask, jsonify, request

app = Flask(__name__)


def create_base_grid():
    xvals = np.linspace(-4, 4, 9)
    yvals = np.linspace(-3, 3, 7)
    xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])
    return xygrid


def colorizer(x, y):
    r = min(1, 1 - y / 3)
    g = min(1, 1 + y / 3)
    b = 1 / 4 + x / 16
    return (r, g, b)


def stepwise_transform(a, points, nsteps=30):
    transgrid = np.zeros((nsteps + 1,) + np.shape(points))
    for j in range(nsteps + 1):
        intermediate = np.eye(2) + j / nsteps * (a - np.eye(2))
        transgrid[j] = np.dot(intermediate, points)
    return transgrid


def make_plots(transarray, color, outdir="png-frames", figuresize=(4, 4), figuredpi=150):
    nsteps = transarray.shape[0]
    ndigits = len(str(nsteps))
    maxval = np.abs(transarray.max())

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # create figure
    plt.ioff()
    fig = plt.figure(figsize=figuresize, facecolor="w")
    for j in range(nsteps):  # plot individual frames
        plt.cla()
        plt.title("Grid da Transformada do espaço x-y para u-v")
        plt.scatter(transarray[j, 0], transarray[j, 1], s=36, c=color, edgecolor="none")
        plt.xlim(1.1 * np.array([-maxval, maxval]))
        plt.ylim(1.1 * np.array([-maxval, maxval]))
        plt.grid(True)
        plt.draw()
        # save as png
        outfile = os.path.join(outdir, "frame-" + str(j + 1).zfill(ndigits) + ".png")
        fig.savefig(outfile, dpi=figuredpi)
    plt.ion()


def make_linear_transformation(x11, x12, x21, x22, steps=10):
    a = np.column_stack([[x11, x12], [x21, x22]])
    xygrid = create_base_grid()
    uvgrid = np.dot(a, xygrid)

    # Map grid coordinates to colors
    colors = list(map(colorizer, xygrid[0], xygrid[1]))

    # Plot x-y grid points
    plt.figure(figsize=(4, 4), facecolor="w")
    plt.scatter(xygrid[0], xygrid[1], s=36, c=colors, edgecolor="none")
    # Set axis limits
    plt.grid(True)
    plt.axis("equal")
    plt.title("Grid Original no espaço x-y")
    plt.savefig("grid-original.png", dpi=150)

    # Plot transformed grid points
    plt.figure(figsize=(4, 4), facecolor="w")
    plt.scatter(uvgrid[0], uvgrid[1], s=36, c=colors, edgecolor="none")
    plt.grid(True)
    plt.axis("equal")
    plt.title("Grid da Transformação no espaço u-v")
    plt.savefig("grid-transformed.png", dpi=150)

    # Apply to x-y grid
    # transform = stepwise_transform(a, xygrid, nsteps=steps)

    # Generate figures
    # make_plots(transform, colors)
    # call(f"cd png-frames && convert -delay {steps} frame-*.png animation.gif", shell=True)
    # call("rm -f png-frames/*.png", shell=True)


def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    jpg_as_text = base64.b64encode(buffer)
    return str(jpg_as_text)


@app.route("/linear_transformation/")
def linear_transformation():
    """ Linear Tranformation Matrix:
          | x y |
          | w z | 
    """
    x = int(request.args.get("x"))
    y = int(request.args.get("y"))
    w = int(request.args.get("w"))
    z = int(request.args.get("z"))

    make_linear_transformation(x, y, w, z)

    image_original = cv2.imread("/workspaces/lineartransformation/grid-transformed.png")
    image_transformed = cv2.imread("/workspaces/lineartransformation/grid-transformed.png")
    data = {
        "original": image_to_base64(image_original),
        "transformed": image_to_base64(image_transformed),
    }
    return jsonify(data)


app.run(host="0.0.0.0", port=5000, debug=True)
