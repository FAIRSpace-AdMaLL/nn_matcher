## nn_matcher

## to select the local feature model
change nn_matcher/src/models/matching.py at line 45

## run the romanwarf demo
./match_pairs_superpointnn.py --viz --input_pairs ./assets/rw.txt --input_dir / --fast_viz True

## run the DNIM demo

### Download the dataset from http://users.umiacs.umd.edu/~hzhou/dnim.html

### create file index (you will need to modify the input and output dictories)
python create_evaluation_file.py

### now you should have files e.g. 00000850.txt in output folder (you need to remove the last line which is empty file).

### define a reference image in each scene (simply choose a night image to rename as ref.jpg)
 
### Then you can run the evluation, use --superpoint tag 'official' or 'dark' (your version)
./match_pairs_superpointnn.py --viz --input_pairs ./assets/00000850.txt --input_dir / --fast_viz True --cache False --superpoint official

### and get output like this:

"[0.015267176, 0.01369863, 0.11805555, 0.071428575, -0.055555556, 0.0, -0.17355372, -0.0625, 0.12962963, -0.14666666, 0.20634921, 0.36585367, -1.5238096, 0.59090906, -0.42105263, -3.6666667, -2.4285715, -0.1, 0.0, 0.22222222, 43.2, 21.222221, 100.6, -0.6, 0.3, 1.3846154, -1.2, 0.7619048, 0.33333334, 0.8333333, 0.23809524, 0.1764706, -0.1, 0.0952381, 0.102564104, -0.20238096, 0.22340426, 0.09677419, 0.21348314, 0.043010753, 0.09090909, 0.20833333, 0.031578947, 0.05154639, -0.2524272, -0.06451613, 0.008196721, 0.29104477, 0.05185185, 0.2977099, 0.33093524, 0.08029197, 0.08, -0.18181819, 0.17333333, 0.08108108, 0.02, 0.2962963, 0.6060606, 0.11111111, -0.18518518, 0.04761905, 0.45454547, -0.85714287, -1.0, -0.5, -2.0, -0.14285715, 2.2222223, -0.2, 20.666666, 18.5, 56.6, 0.1875, 0.3888889, 1.2142857, -0.42857143, -0.5625, -0.61538464, 0.05263158, 0.0, 0.53333336, 0.41975307, -0.15463917, 0.46067417, 0.1978022, -0.20454545, 0.0989011, -0.05263158, 0.3043478, 0.15463917, -0.22105263, 0.09090909, -0.008333334, 0.4051724]
overall error: 13.160441476205607"


