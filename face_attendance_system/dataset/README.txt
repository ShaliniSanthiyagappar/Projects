dataset/ -- YOUR KNOWN FACES GO HERE
====================================

Create ONE folder per person. The FOLDER NAME is the name that will be
shown on screen and written to the database, so name folders carefully.

    dataset/
    |-- Alice_Smith/
    |   |-- photo1.jpg
    |   |-- photo2.jpg
    |   `-- photo3.jpg
    `-- Raj_Patel/
        `-- selfie.png

Photo tips (3-10 photos per person):
  * looking at the camera, face clearly visible, no sunglasses
  * vary distance and angle a little (front, slight left, slight right)
  * good, even lighting - avoid heavy shadows and motion blur
  * JPG / JPEG / PNG / WEBP are all accepted

After adding photos run:

    python dataset_builder.py

Re-run it any time you add people - unchanged photos are reused, so it is
fast. Use `python dataset_builder.py --force` to rebuild from scratch.