#!/bin/sh
pandoc -f markdown --defaults docs/config_pdf.yaml -o LESSON.pdf LESSON.md

