#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import gen_scene_dicts, image_names_from_transition, build_graph_scenes, check_existing_transitions
from aa_utils.plot import plot_scene_sequence

from dotenv import load_dotenv; load_dotenv()
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='', dest='scene_sequence')
# args = parser.parse_args()
args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

scene_dir = pjoin(gdrive_basedir, args.song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

scene_sequence_name = "scene_sequence" if args.scene_sequence == '' else "scene_sequence_{}".format(args.scene_sequence)
fp_scene_sequence = os.path.join(os.getenv('repo_dir'), 'song_meta', args.song, '{}.csv'.format(scene_sequence_name))
scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)['scene'].values.tolist()

# Make a mapping from file to folder name for each scene folder in scene dir
# We truncate here as transition folders are truncated to 4 digits...
scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence, truncate_digits=None)

# %%

scene_dict


import reportlab

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

# %%

# use reportlab to generate a pdf, with a page for each key in scene_dict, and a table of images for each value

# https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/

from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Table, TableStyle
from reportlab.lib.utils import ImageReader

styles = getSampleStyleSheet()

def generate_color_scheme(filepaths):
    # create an array to store the dominant colors from each image
    colors = np.zeros((len(filepaths), 3))

    # extract the dominant color from each image
    for i, fp in enumerate(filepaths):
        
        img = PILImage.open(fp)

        img = img.convert('P', palette=PILImage.ADAPTIVE, colors=4)
        colors[i] = img.getpalette()[:3]

    # create a new image with the dominant colors
    box_width=50
    box_height=20

    img_width = len(filepaths) * box_width
    img_height = box_height
    color_scheme = PILImage.new('RGB', (img_width, img_height), (255, 255, 255))
    for i, color in enumerate(colors):
        x0 = i * box_width
        x1 = (i + 1) * box_width
        y0 = 0
        y1= box_height
        color = tuple(int(c) for c in color)
        color_box = PILImage.new('RGB', (box_width, box_height), color)
        color_scheme.paste(color_box, (x0, y0, x1, y1))

    return color_scheme



def build_scene_sequence_story(scene_dict, story=[]):

    for scene in scene_dict:

        scene_images = scene_dict[scene]
        scene_images = [s.replace('-', '_') for s in scene_images]
        # replace - with _ 
        # scene_images = [s.replace(' ', '\ ') for s in scene_images]

        # add the key as a heading
        story.append(Paragraph("Scene: {}".format(scene), styles['Heading1']))

        # create a table with 2 columns and as many rows as needed for the images
        # the images have 16/9 aspect ratio, and scale them to 4 inches wide
        filepaths = [pjoin(scene_dir, scene, image_name + '.png') for image_name in scene_images]
        images = [Image(filepath, width=4*inch, height=4*(9/16)*inch) for filepath in filepaths]

        # assume that 'color_scheme' is a Pillow image object
        color_scheme = generate_color_scheme(filepaths)

        # save the scene color theme to a file, have to save each separately
        fp_color_scheme = pjoin(color_scheme_output_dir, 'color_scheme_{}.png'.format(scene))
        color_scheme.save(fp_color_scheme)
        color_scheme_flowable = Image(fp_color_scheme, width=4*inch, height=0.5*inch)

        story.append(color_scheme_flowable)

        table_data = []
        row = []
        for i, image_name in enumerate(scene_images):
            # add each image to the table
            # img = Image(filename, width=2*inch, height=2*inch)
            filepath = filepaths[i]

            sub_table_data = [
                [Paragraph(image_name, styles['Normal'])],
                [images[i]]
            ]
            sub_table = Table(sub_table_data)
            row.append(sub_table)

            # start a new row after every 3 images``
            if (i+1) % 2 == 0:
                table_data.append(row)
                row = []

        # add the last row if it's not empty
        if row:
            table_data.append(row)

        # add the table to the story
        table = Table(table_data)



        story.append(table)

            # add a page break before each new key
        story.append(PageBreak())

    return story


output_pdf_path = os.path.join(gdrive_basedir, args.song, 'story', 'scene_report.pdf')

#TODO: Can we make a flowable image from PIL image without saving to disk?
color_scheme_output_dir = 'output/color_schemes'
if not os.path.exists(color_scheme_output_dir):
    os.makedirs(color_scheme_output_dir)

# keep a random selecton of 3 scenes
# scene_dict = {k:scene_dict[k] for k in np.random.choice(list(scene_dict.keys()), 3)}

doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)

story = build_scene_sequence_story(scene_dict)

#TODO: add to the beginning of story a table of scene_dict, with a second column of color schemes

# build the PDF
doc.build(story)