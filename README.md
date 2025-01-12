# PianoChord
Converts a piano melody score to a three-line score, which makes piano accompaniments (input file formats are png, jpg, and midi.)
## Components
download.py : runs web crawling to download learning data from Akbobada   
toxml.py : converts file formats to midi like xml, musicxml. so files can be read   
read.py : tokenizes, preprocesses for learning   
learn.py : runs transformer-based-learning for each first model (travel, bass)   
ac_learn.py : processes learning for each second model   
test.py : tests
## Models
It uses two models. The first model generates a three-line piano score from a melody-only sheet. However, it only generates measures that contain melody notes, whereas many real scores include measures that only have accompaniment (i.e., empty melody measures). The second model complements the first by taking those partially generated scores and filling in the empty measures, effectively learning how to complete the entire sheet.
