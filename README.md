# Destruction-Detection-in-Satellite-Imagery
Detection of destruction in satellite imagery

### 1. Training
To train model just enter this command:

!python train.py --trainfeatures_filename trainfeatures.pickle --epochs 500

Recommended iteration=500

### 2. Retraining
To retrain our network put this command:

!python retraining.py --model_name Model1_AttentionNetwork_500.h5 --trainfeatures_filename trainfeatures.pickle --epochs 500

It will retrain the first model using Hard negative mining approach to 500 epochs

### 3. Testing
Before testing the model, there is need to generate segmentation masks.To generate mask enter these two commands one by one:

!python segmentation.py --model_name Model1_AttentionNetwork_500.h5 --test_path Data/test --apply_CRF no
!python segmentation.py --model_name Model2_retrain_AttentionNetwork_500.h5 --test_path Data/test --apply_CRF yes

Now to see Testing results. Enter this command:

!python testing.py --model1_name Model1_AttentionNetwork_500.h5 --model_retrain_name Model2_retrain_AttentionNetwork_500.h5 --features_filename testfeatures.pickle



