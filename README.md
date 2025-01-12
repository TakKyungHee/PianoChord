# PianoChord
Converts a piano melody score to a three-line score. (input file formats are png, jpg, and midi.)
## Components
download.py : web crawling to download learning data from Akbobada   
toxml.py : converting file formats to midi like xml, musicxml. so files are able to be read   
read.py : tokenizing, preprocessing for learning   
learn.py : transformer-based-learning for each first model (travel, bass)   
ac_learn.py : learning for each second model   
test.py : testing
## Models
It uses two models. The first model generates a three-line piano score from a melody-only sheet. However, it only generates measures that contain melody notes, whereas many real scores include measures that only have accompaniment (i.e., empty melody measures). The second model complements the first by taking those partially generated scores and filling in the empty measures, effectively learning how to complete the entire sheet.
