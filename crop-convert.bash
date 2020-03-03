#!/bin/bash
rm -f *crop*pdf
cd pdfs/
pwd
for file in *.pdf
do
    pdfcrop --margins '60 0 40 0' $file 1.pdf
    convert -density 200 1.pdf -quality 90 1.png
    s=`echo ${file} |cut -d '.' -f 1`
    mv 1.png ../images/$s.png
    rm -f 1.pdf
    echo $file
done
