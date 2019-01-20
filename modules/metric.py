import editdistance
import numpy as np
import pandas as pd
from modules.beam_search import decode

class meter:
    def __init__(self, blank=None, source_phn=48):
        self.guessed_labels = []
        self.target_labels = []
        self.blank = blank
        self.source_phn = source_phn

        # Create PHN-48 to PHN-39 mapping dict
        phn_map_48_39 = pd.read_csv('./data/phn_map_48_39.csv')
        self.dict_48_39_int = {}
        for idx, x in enumerate(phn_map_48_39['phn-48-int']):
            self.dict_48_39_int[x + 1] = int(phn_map_48_39['phn-39-int'][idx]) + 1  # Create PHN-48 to PHN-39 conversion dict

        # Create PHN-61 to PHN-39 mapping dict
        phn_map_61_39 = pd.read_csv('./data/phn_map_61_48_39.csv')
        self.dict_61_39_int = {}
        for idx, x in enumerate(phn_map_61_39['phn-61-int']):
            self.dict_61_39_int[x + 1] = int(phn_map_61_39['phn-39-int'][idx]) + 1  # Create PHN-48 to PHN-39 conversion dict

    def extend_guessed_labels(self, prediction, decoder='greedy', beam_width=10, source_phn=48):
        guessed_batch_labels = self.convert_prediction_to_transcription(prediction, self.blank, decoder=decoder, beam_width=beam_width, source_phn=source_phn)
        self.guessed_labels.extend(guessed_batch_labels)

        return self.guessed_labels

    def extend_target_labels(self, bY, b_lenY):
        # TODO remove the easier batch labels step once the hdf5 are fixed
        easier_batch_labels = self.convert_from_ctc_to_easy_labels(bY, b_lenY)  # ease access to warp-ctc labels
        target_batch_labels = [vec2str(label) for label in easier_batch_labels]  # prepare string
        self.target_labels.extend(target_batch_labels)

        return self.target_labels

    def get_metrics(self):
        phone_error_rate = calculate_error_rates(self.target_labels, self.guessed_labels)
        self.guessed_labels = []
        self.target_labels = []
        return phone_error_rate

    def get_metrics_preserve(self):
        phone_error_rate = calculate_error_rates(self.target_labels, self.guessed_labels)
        return phone_error_rate

    def clear(self):
        self.guessed_labels = []
        self.target_labels = []

    def greedy_decoder_map_phn(self, sample_prediction):
        guess_vec = np.argmax(sample_prediction, axis=1)
        return guess_vec

    def beam_decoder_map_phn(self, sample_prediction, blank, beam_width):
        guess_vec = decode(sample_prediction, beam_size=beam_width, blank=blank)
        guess_vec = np.asarray(guess_vec[0], dtype=np.int32)
        return guess_vec


    def convert_prediction_to_transcription(self, prediction, blank, decoder, beam_width, source_phn=48):
        # Prediction input : Time X Batch X Features 3D matrix
        prediction = prediction.transpose([1, 0, 2])
        if decoder == 'greedy':
            guessed_labels = [self.greedy_decoder_map_phn(phrase) for phrase in prediction]
        elif decoder == 'beam':
            guessed_labels = [self.beam_decoder_map_phn(phrase, blank, beam_width) for phrase in prediction]

        guessed_labels_final = []
        for guess_vec in guessed_labels:
            if source_phn == 61:
                # Remove label 'q' ()
                guess_vec[np.where(guess_vec == 58)] = 0
                for i in range(0, guess_vec.shape[0]):
                    if guess_vec[i] != 0:
                        guess_vec[i] = self.dict_61_39_int[guess_vec[i]]
            elif source_phn == 48:
                for i in range(0, guess_vec.shape[0]):
                    if guess_vec[i] != 0:
                        guess_vec[i] = self.dict_48_39_int[guess_vec[i]]
            guess_vec_elim = vec2str(eliminate_duplicates_and_blanks(guess_vec, blank))
            guessed_labels_final.append(guess_vec_elim)

        return guessed_labels_final

    def convert_from_ctc_to_easy_labels(self, bY, lenY):

        curr_idx = 0
        curr_label = 0
        labels = []
        while curr_idx < len(bY):
            curr_len = lenY[curr_label]
            label_list = bY[curr_idx:curr_idx + curr_len]
            labels.append([item for item in label_list])
            curr_idx += curr_len
            curr_label += 1
        return labels

def vec2str(guess_vec):
    guessed_label = '-'.join([str(item) for item in guess_vec])

    return guessed_label


def eliminate_duplicates_and_blanks(guess_vec, blank):

    rv = []
    # Remove duplicates
    for item in guess_vec:
        if (len(rv) == 0 or item != rv[-1]):
            rv.append(item)

    # Remove blanks (warp ctc label: label 0, tensorflow: last label)
    final_rv = []
    for item in rv:
        if item != blank:
            final_rv.append(item)
    return final_rv


def calculate_error_rates(target_labels, guessed_labels):

    # Phone error rate
    chars_wrong = 0
    total_chars = 0
    for idx, target in enumerate(target_labels):
        guess_chars = guessed_labels[idx].split('-')
        target_chars = target.split('-')
        errors = int(editdistance.eval(target_chars, guess_chars))
        chars_wrong += errors
        total_chars += len(target_chars)
    CER = float(chars_wrong) / total_chars

    return CER

