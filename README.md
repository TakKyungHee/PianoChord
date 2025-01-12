# PianoChord
Converts piano melody sheet music to piano 3 lines sheet music. (input file formats are png, jpg, and midi.)
## Components
download.py : web crawling to download learning data from Akbobada   
toxml.py : converting file formats to midi like xml, musicxml. so files are able to be read   
read.py : tokenizing, preprocessing for learning   
learn.py : transformer-based-learning for each first model(travel, bass)   
ac_learn.py : learning for each second model   
test.py : for testing
## models
It uses two models. First one makes piano 3 lines sheets from melody sheets. but it only makes sheets from non-empty measure of melody while many sheets have measures in which only accompaniments exist(melody measures are empty). Second one complements it by making sheets from the sheets first one made to complete sheets. It learn how to fill in the empty measures.
