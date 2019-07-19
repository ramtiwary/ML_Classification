from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii',
                                   stop_words='english', min_df=0.0005, sublinear_tf=True)
codebook_vectorizer = tfidf_vectorizer.fit(instances_processed)

PARAMS_TRAIN = {
    'kernel': ('linear', 'rbf'),
    'C': [1, 10, 1e2, 1e3],
    'gamma': [0.00001, 0.0001, 0.001, 0.005],
     }



def train_custom_nn(self, vec_size, n_classes, l2_strength = 1e-5):
     text_input = Input(shape=(vec_size,), name ='text_input', sparse = True)
     text_layer = Dense(vec_size, activation='relu',
                        kernel_regularizer = l2(l2_strength ))(text_input)

     drop = Dropout(0.5)(text_layer)
     final_output = Dense(n_classes, activation ='softmax')(drop)
     final_model =  Model(inputs=[text_input], outputs=[final_output])

     return final_model


check_pointer = ModelCheckpoint(filepath=self.FILE_WEIGHTS,verbose=1,save_best_only=True)
early_stopper = EarlyStopping(monitor='val_loss',patience=5)
