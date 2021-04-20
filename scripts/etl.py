"""Pre-process data.

Convert both train and test data from audio to spectrograms.
Create a list breaking up training data into train/test sets.
Pickle the data and save to disk.
"""

import os
import pickle
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
import librosa
from utility import parse_config, set_logger

@click.command()
@click.argument("config_file", type=str, default="config.yaml")

def etl(config_file):


    ##################
    # configure logger
    ##################
    logger = set_logger("./log/etl.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)
    reprocess_audio = config['etl']['reprocess_audio']
    dev_mode = config['etl']['dev_mode']

    #get the root directory of the repository
    dirname = os.path.dirname(os.path.dirname(__file__))
    raw_data_file = os.path.join(dirname, config['etl']['raw_data_file'])
    training_files_path = os.path.join(dirname, config['etl']['training_files_path'])
    processed_path = os.path.join(dirname, config['etl']['processed_path'])
    test_files_list = os.path.join(dirname, config['etl']['test_files_list'])
    test_files_path = os.path.join(dirname, config['etl']['test_files_path'])
    train_test_split_method = config['etl']['train_test_split_method']
    random_state = config['etl']['random_state']
    test_size = config['etl']['test_size']
    length_of_section = config['etl']['length_of_section']
    sr = config['etl']['sr']
    frame_size = config['etl']['frame_size']
    hop_size = config['etl']['hop_size']
    aug_1_yn = config['etl']['aug_1_yn']
    aug_1_addnoise = config['etl']['aug_1_addnoise']
    aug_1_timeshift = config['etl']['aug_1_timeshift']
    aug_1_pitchfactor = config['etl']['aug_1_pitchfactor']
    aug_1_speedfactor = config['etl']['aug_1_speedfactor']
    aug_2_yn = config['etl']['aug_2_yn']
    aug_2_addnoise = config['etl']['aug_2_addnoise']
    aug_2_timeshift = config['etl']['aug_2_timeshift']
    aug_2_pitchfactor = config['etl']['aug_2_pitchfactor']
    aug_2_speedfactor = config['etl']['aug_2_speedfactor']
    aug_3_yn = config['etl']['aug_3_yn']
    aug_3_addnoise = config['etl']['aug_3_addnoise']
    aug_3_timeshift = config['etl']['aug_3_timeshift']
    aug_3_pitchfactor = config['etl']['aug_3_pitchfactor']
    aug_3_speedfactor = config['etl']['aug_3_speedfactor']
    aug_4_yn = config['etl']['aug_4_yn']
    aug_4_addnoise = config['etl']['aug_4_addnoise']
    aug_4_timeshift = config['etl']['aug_4_timeshift']
    aug_4_pitchfactor = config['etl']['aug_4_pitchfactor']
    aug_4_speedfactor = config['etl']['aug_4_speedfactor']
    test_segment_overlap = config['etl']['test_segment_overlap']
    logger.info(f"config: {config['etl']}")
    ##################
    # Data transformation
    ##################
    logger.info("----------------Start data transformation-------------------")
    ##################
    # Get lists of training and testing audio files
    ##################
    #there is a csv file which contains the filenames of the training examples,
    #and the times at which the applicable section appears.
    #Load that file to a Pandas dataframe
    train_tp = pd.read_csv(raw_data_file)
    #if we are in development mode, shorten the dataframe
    #so this doesn't take a million years
    if dev_mode:
        train_tp = train_tp[1:100]
    #there is also a csv file that lists all the files we need to test.
    #load into a pandas dataframe
    test_files = pd.read_csv(test_files_list)
    if dev_mode:
        test_files = test_files[1:100]
    ##################
    # Transform and Augment Training Data
    ##################

    """
    this function loops through the training examples in train_tp
    (or any other pandas dataframe), converts them into mel spectrograms, 
    and saves them, along with label data and row id data as a pickle file.
    If this is being run with reprocess_audio==False, it just loads the old
    pickle files rather than recreate them.
    This is a function, rather than a single for loop so it can be called 
    multiple times on the same set of data to allow for data augmentation
    This is defined as an inner function so it can share variables with the 
    etl function.
    """
    def process_a_set_of_training_data(
            reprocess_audio=False,
            input=train_tp,
            label_input_column_name='species_id',
            #filename_column_name='recording_id',
            output_lsr_keys=['labels', 'spectrograms', 'row_ids'],
            audio_file_path=training_files_path,
            addnoise=0,
            timeshift=0,
            pitchfactor=0,
            speedfactor=0
            ):
        #create a dictionary of empty lists
        empty_lists = [[], [], []]
        outputs = dict(zip(output_lsr_keys, empty_lists))
        if reprocess_audio == True:
            #process all the files mentioned in your dataframe
            for row_num in range(0, len(input)):
                #for each training example, get the audio file name, and the
                #start and end times of the section of interest
                section_start = float(train_tp.iloc[row_num]['t_min'])
                section_end = float(train_tp.iloc[row_num]['t_max'])
                section_contained_in_filename = train_tp.iloc[row_num]['recording_id']
                #get the label and add to the list
                outputs[output_lsr_keys[0]].append(input.iloc[row_num][label_input_column_name])
                #create a spectrogram and add to the list
                outputs[output_lsr_keys[1]].append(
                    get_mel_spectrogram(
                        row_num=row_num,
                        start=section_start,
                        end=section_end,
                        filename=section_contained_in_filename,
                        path=training_files_path,
                        sr=sr,
                        length=length_of_section,
                        frame_size=frame_size,
                        hop_size=hop_size,
                        mode='Train',
                        test_clip_num=0,
                        addnoise=0,
                        timeshift=0,
                        pitchfactor=0,
                        speedfactor=0))
                #add row ids to list
                outputs[output_lsr_keys[2]].append(row_num)
            #save the resulting lists to picklefiles
            with open(str(processed_path)+'\\'+output_lsr_keys[0]+'.pickle', 'wb') as pickle_file:
                pickle.dump(outputs[output_lsr_keys[0]], pickle_file, pickle.HIGHEST_PROTOCOL)
            with open(str(processed_path)+'\\'+output_lsr_keys[1]+'.pickle', 'wb') as pickle_file:
                pickle.dump(outputs[output_lsr_keys[1]], pickle_file, pickle.HIGHEST_PROTOCOL)
            with open(str(processed_path)+'\\'+output_lsr_keys[2]+'.pickle', 'wb') as pickle_file:
                pickle.dump(outputs[output_lsr_keys[2]], pickle_file, pickle.HIGHEST_PROTOCOL)
            return outputs[output_lsr_keys[0]], outputs[output_lsr_keys[1]],\
                outputs[output_lsr_keys[2]]

        if reprocess_audio == False:
            #open up the existing pickle files and load the lists into variables
            with open(str(processed_path)+'\\'+output_lsr_keys[0]+'.pickle', 'rb') as pickle_file:
                outputs[output_lsr_keys[0]] = pickle.load(pickle_file)
            with open(str(processed_path)+'\\'+output_lsr_keys[1]+'.pickle', 'rb') as pickle_file:
                outputs[output_lsr_keys[1]] = pickle.load(pickle_file)
            with open(str(processed_path)+'\\'+output_lsr_keys[2]+'.pickle', 'rb') as pickle_file:
                outputs[output_lsr_keys[2]] = pickle.load(pickle_file)
            return outputs[output_lsr_keys[0]], outputs[output_lsr_keys[1]],\
                outputs[output_lsr_keys[2]]

    #run the process_a_set_of_training_data function at least once
    labels, spectrograms, row_ids = process_a_set_of_training_data(
        reprocess_audio=reprocess_audio,
        output_lsr_keys=['labels', 'spectrograms', 'row_ids'])
    #if we want to do any augmentation, run it for each of the 4 optional augmentations as well
    if aug_1_yn:
        labels_aug1, spectrograms_aug1, row_ids_aug1 = process_a_set_of_training_data(
            reprocess_audio=reprocess_audio,
            output_lsr_keys=['labels_aug1', 'spectrograms_aug1', 'row_ids_aug1'],
            addnoise=aug_1_addnoise, timeshift=aug_1_timeshift,
            pitchfactor=aug_1_pitchfactor, speedfactor=aug_1_speedfactor)

    if aug_2_yn:
        labels_aug2, spectrograms_aug2, row_ids_aug2 = process_a_set_of_training_data(
            reprocess_audio=reprocess_audio,
            output_lsr_keys=['labels_aug2', 'spectrograms_aug2', 'row_ids_aug2'],
            addnoise=aug_2_addnoise, timeshift=aug_2_timeshift,
            pitchfactor=aug_2_pitchfactor, speedfactor=aug_2_speedfactor)

    if aug_3_yn:
        labels_aug3, spectrograms_aug3, row_ids_aug3 = process_a_set_of_training_data(
            reprocess_audio=reprocess_audio,
            output_lsr_keys=['labels_aug3', 'spectrograms_aug3', 'row_ids_aug3'],
            addnoise=aug_3_addnoise, timeshift=aug_3_timeshift,
            pitchfactor=aug_3_pitchfactor, speedfactor=aug_3_speedfactor)

    if aug_4_yn:
        labels_aug4, spectrograms_aug4, row_ids_aug4 = process_a_set_of_training_data(
            reprocess_audio=reprocess_audio,
            output_lsr_keys=['labels_aug4', 'spectrograms_aug4', 'row_ids_aug4'],
            addnoise=aug_4_addnoise, timeshift=aug_4_timeshift,
            pitchfactor=aug_4_pitchfactor, speedfactor=aug_4_speedfactor)

    ##################
    # train test split & Export
    ##################
    #logger.info(f"train: {train.shape}")

    if train_test_split_method == 'Simple':
        #this just creates a train test split on the unaugmented data.
        #The augmented data is not used.
        train_rowids, test_rowids = custom_train_test_splitter(
            method='Simple', random_state=random_state, test_size=test_size,
            row_ids_list=row_ids)

        with open(str(processed_path)+'\\'+'train_rowids.pickle', 'wb') as pickle_file:
            pickle.dump(train_rowids, pickle_file, pickle.HIGHEST_PROTOCOL)

        with open(str(processed_path)+'\\'+'test_rowids.pickle', 'wb') as pickle_file:
            pickle.dump(test_rowids, pickle_file, pickle.HIGHEST_PROTOCOL)


    #delete your test objects
    del labels
    del spectrograms
    del row_ids
    
    ##################
    # Tranform Test Data (i.e. data used for evaluation in competition)
    ##################

    test_segments = get_test_segments(overlap=test_segment_overlap, sr=sr)


    if reprocess_audio is False:
    #     with open('labels_test.pickle', 'rb') as f:
    #         labels = pickle.load(f)
        with open(str(processed_path)+'\\'+'spectrograms_test.pickle', 'rb') as pickle_file:
            spectrograms_test = pickle.load(pickle_file)
        with open(str(processed_path)+'\\'+'row_ids_test.pickle', 'rb') as pickle_file:
            row_ids_test = pickle.load(pickle_file)
        with open(str(processed_path)+'\\'+'clip_segments_test.pickle', 'rb') as pickle_file:
            clip_segments_test = pickle.load(pickle_file)
        with open(str(processed_path)+'\\'+'clips_test.pickle', 'rb') as pickle_file:
            clips_test = pickle.load(pickle_file)

    else:
        row_ids_test = []
        clips_test = []
        clip_segments_test = []
        spectrograms_test = []

        for i in range(0, len(test_files)):
            test_file_name = test_files.iloc[i]['recording_id']
            for j in range(0, len(test_segments)):
                clips_test.append(test_file_name)
                clip_segments_test.append(j)
                #create a spectrogram and add to the list
                spectrograms_test.append(get_mel_spectrogram(
                    row_num=i, start=0, end=0, filename=test_file_name,
                    path=test_files_path, sr=sr, length=length_of_section,
                    frame_size=frame_size, hop_size=hop_size, mode="Test",
                    test_clip_num=j, addnoise=0, timeshift=0, pitchfactor=0,
                    speedfactor=0, test_segments=test_segments))
                row_ids_test.append(i)
        with open(str(processed_path)+'\\'+'spectrograms_test.pickle', 'wb') as pickle_file:
            pickle.dump(spectrograms_test, pickle_file, pickle.HIGHEST_PROTOCOL)

        with open(str(processed_path)+'\\'+'row_ids_test.pickle', 'wb') as pickle_file:
            pickle.dump(row_ids_test, pickle_file, pickle.HIGHEST_PROTOCOL)

        with open(str(processed_path)+'\\'+'clip_segments_test.pickle', 'wb') as pickle_file:
            pickle.dump(clip_segments_test, pickle_file, pickle.HIGHEST_PROTOCOL)

        with open(str(processed_path)+'\\'+'clips_test.pickle', 'wb') as pickle_file:
            pickle.dump(clips_test, pickle_file, pickle.HIGHEST_PROTOCOL)

    logger.info("End data transformation")





##################
# Define Helper Functions used in Transformation
##################

"""

The following 3 functions will be used for data augmentation purposes. 
i.e. Creating new, slightly different versions of the examples which will be 
added to the training set to reduce overfitting and make the model more robust

"""
def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def pitch_shift(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def time_stretch(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


"""
The data provided consists of 60 second clips, which at times, contain audio which
represents one of the 24 species we would like to identify. The functions below are used
to convert those files into the shorter samples we will use for training
"""

"""
The training and test datasets contains a number of one-minute audio (.flac) files.
In the training set, each file one or more "sections" that are of interest.
Each of these is annotated by a single row
in the train_tp, or train_fp dataframes.
This function gets the audio for a given section, based on the row number from
the dataframe.
"""

def get_flac_section(row_num, start, end, filename, path, sr, length,
                     addnoise=0, timeshift=0, pitchfactor=0, speedfactor=0):
    #this function is called by the get_mel_spectrogram function while processing training data
    #find the midpoint of the training section as defined in the data provided,
    #and then re-assign the start and end variables to create a section of length
    #'length', starting length/2 before the midpoint
    midpoint = (start + end) / 2

    #this is used for audio augmentation, shifts the midpoint by a specified length
    midpoint = midpoint + timeshift
    start = midpoint - (length / 2)
    end = midpoint + (length / 2)

    #if this ends up starting before 0, then start the section at zero instead.
    if start < 0:
        start = 0
        end = length
    #if this ends up ending after 60 seconds, start at 60-length_of_section seconds
    if end > 60:
        start = 60 - length
        end = 60

    #get the file in question
    flacdata, samplerate = librosa.load(path+'/'+filename+'.flac', sr)

    #calculate the start and end coordinates
    startsample = round((samplerate * start), None)
    endsample = round((samplerate * end), None)
    fd = flacdata[startsample:endsample]

    if addnoise != 0:
        fd = add_noise(fd, addnoise)

    if pitchfactor != 0:
        fd = pitch_shift(fd, sampling_rate=sr, pitch_factor=pitchfactor)

    if speedfactor != 0:
        #whole numbers make it go faster, fractions of 1 make it go slower
        #stretch the data
        fd = time_stretch(fd, speed_factor=speedfactor)
        #clip it so it's the same size as the others
        fd = fd[0:(length*sr)]

    return fd, samplerate
    #return startsample, endsample

def get_mel_spectrogram(row_num, start, end, filename, path, sr, length,
                        frame_size, hop_size, mode='Train', test_clip_num=0,
                        addnoise=0, timeshift=0, pitchfactor=0, speedfactor=0,
                        test_segments=[]):

    if mode == 'Train':
        y, sr = get_flac_section(row_num=row_num, start=start, end=end,
                                 filename=filename, path=path, sr=sr,
                                 length=length, addnoise=addnoise,
                                 timeshift=timeshift, pitchfactor=pitchfactor,
                                 speedfactor=speedfactor)
    if mode == 'Test':
        y, sr = get_flac_section_for_test(row_num=row_num, filename=filename,
                                          which_segment=test_clip_num, path=path,
                                          sr=sr, test_segments=test_segments)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=int(sr),
                                                 n_fft=int(frame_size),
                                                 hop_length=int(hop_size))
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max) # Converting to decibals

    spectrogram = resize(spectrogram, (224, 400))

    # Normalize to 0...1
    spectrogram = spectrogram - np.min(spectrogram)
    spectrogram = spectrogram / np.max(spectrogram)
    return spectrogram

def get_test_segments(overlap, sr):
    #The test data is in 60 second files. We need to split them up into 6
    #second segments that match the train data.
    #define how your test data should be split up.
    #Optionally, these segments can overlap.
    test_segments = pd.DataFrame(columns=['start', 'end'])

    if overlap:
        test_segments['start'] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                                  33, 36, 39, 42, 45, 48, 51, 54]
        test_segments['end'] = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36,
                                39, 42, 45, 48, 51, 54, 57, 60]
    else:
        test_segments['start'] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
        test_segments['end'] = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

    test_segments['start'] = [x * sr for x in test_segments['start']]
    test_segments['end'] = [x * sr for x in test_segments['end']]
    return test_segments

def get_flac_section_for_test(row_num, filename, which_segment, path, sr, test_segments):
    #get the file in question
    flacdata, samplerate = librosa.load(str(path)+'//'+filename+'.flac', sr)

    start = test_segments.iloc[which_segment]['start']
    end = test_segments.iloc[which_segment]['end']

    fd = flacdata[start:end]
    return fd, samplerate


def custom_train_test_splitter(method, random_state, test_size, row_ids_list):
    if method == 'Simple':
        train, test = train_test_split(row_ids_list, test_size=test_size, random_state=random_state)
    return train, test


if __name__ == "__main__":
    etl()
