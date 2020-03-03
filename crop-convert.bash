#!/bin/bash
rm -f *crop*pdf
cd images/
pwd
for file in *.pdf
do
    pdfcrop --margins '100 0 100 0' $file 1.pdf
    convert -density 100 1.pdf -quality 90 1.png
    s=`echo ${file} |cut -d '.' -f 1`
    mv 1.png ../images/$s.png
    rm -f $file 1.pdf
    echo $file
done
