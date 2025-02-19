#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_image(args: argparse.Namespace):
    """Detects empty layers and removes them from the manifest and config.hash json file"""
    
    logger.info("Preparing working directory...")
    path = args.image.replace(".tar", "")
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
    subprocess.check_output(["tar", "-xf", args.image, "-C", path])
    curdir = os.getcwd()
    os.chdir(path)
    
    logger.info("Gathering data from compressed image files...")
    with open("manifest.json") as manifest:
        manifest_json = json.load(manifest)[0]
        conf_name = manifest_json["Config"]
        layers = manifest_json["Layers"]
    
    with open(conf_name) as conf:
        conf_json = json.load(conf)
        all_layers = conf_json["history"]
        layers_indexes = [
            ind for ind, layer in enumerate(all_layers) if "empty_layer" not in layer
        ]
    
    layers_data = dict(zip(layers, layers_indexes))
    changed = False
    
    for layer in layers[:]:
        with open(layer, "rb") as layer_content:
            bytes_layer_content = layer_content.read()
            if not any(bytes_layer_content):
                logger.info("%s is an empty archive" % layer)
                ind = layers_data[layer]
                logger.info("Coming from: %s" % all_layers[ind])
                all_layers.pop(ind)
                hash_layer256 = hashlib.sha256(bytes_layer_content).hexdigest()
                logger.debug("With the hash: %s" % hash_layer256)
                conf_json["diff_ids"].remove(f"sha256:{hash_layer256}")
                layers.remove(layer)
                dir_path = layer.replace("/layer.tar", "")
                shutil.rmtree(dir_path)
                changed = True
    
    output = "../%s" % args.output
    if changed:
        logger.info("Updating image to new changes")
        with open("manifest.json", "w") as manifest:
            json.dump([manifest_json], manifest)
        with open(conf_name, "w") as conf:
            json.dump(conf_json, conf)
        subprocess.check_output(["tar", "-cf", output, *os.listdir()])
    else:
        logger.info("Nothing to update")
        shutil.copy("../%s" % args.image, output)
    
    os.chdir(curdir)
    shutil.rmtree(path)
    logger.info("Cleaning ended successfully")

def main():
    parser = argparse.ArgumentParser(description="Clean empty layers from Docker image")
    parser.add_argument("image", help="Input Docker image (.tar file)")
    parser.add_argument("output", help="Output Docker image file")
    args = parser.parse_args()
    clean_image(args)

if __name__ == "__main__":
    main()