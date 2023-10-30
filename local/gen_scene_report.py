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
parser.add_argument('--ss', default='scene_sequence_kv3', dest='scene_sequence')
# args = parser.parse_args()
args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

scene_dir = pjoin(gdrive_basedir, args.song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

fp_scene_sequence = os.path.join(gdrive_basedir, args.song, 'prompt_data', '{}.csv'.format(args.scene_sequence))
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

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak, Table, TableStyle

styles = getSampleStyleSheet()

def build_pdf(scene_dict, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    story = []

    for scene in scene_dict:

        scene_images = scene_dict[scene]
        scene_images = [s.replace('-', '_') for s in scene_images]
        # replace - with _ 
        # scene_images = [s.replace(' ', '\ ') for s in scene_images]


        # add a page break before each new key
        story.append(PageBreak())

        # add the key as a heading
        story.append(Paragraph(scene, styles['Heading1']))

        # create a table with 2 columns and as many rows as needed for the images
        # the images have 16/9 aspect ratio, and scale them to 4 inches wide
        table_data = []
        row = []
        for i, image_name in enumerate(scene_images):
            # add each image to the table
            # img = Image(filename, width=2*inch, height=2*inch)
            filepath = pjoin(scene_dir, scene, image_name + '.png')

            sub_table_data = [
                [Paragraph(image_name, styles['Normal'])],
                [Image(filepath, width=4*inch, height=4*(9/16)*inch)]
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

    # build the PDF
    doc.build(story)


output_pdf_path = os.path.join(gdrive_basedir, args.song, 'story', 'scene_report.pdf')

# # only keep the first three scenes for now
# scene_dict = {k:scene_dict[k] for k in list(scene_dict.keys())[:3]}

story = build_pdf(scene_dict, output_pdf_path)