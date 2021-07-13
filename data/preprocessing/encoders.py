#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data.based.encoder_enum import EncoderTypes


class Encoders:
    def __init__(self, cdg):
        self._cfg = cdg

    def __get_encoder(self, encoder_type, col):
        """
        initialize encoder object

        :param encoder_type:
        :param col: columns that you want to encode.Iit is used for category_encoders package
        :return: encoder obj, encoder name
        """
        le = None
        le_name = None
        if encoder_type == EncoderTypes.LABEL:
            le = LabelEncoder()
            le_name = 'label_encoding'
        elif encoder_type == EncoderTypes.ORDINAL:
            le = OrdinalEncoder(cols=[col])
            le_name = 'ordinal_encoding'
        elif encoder_type == EncoderTypes.ONE_HOT:
            le = OneHotEncoder(cols=[col])
            le_name = 'one_hot_encoding'
        elif encoder_type == EncoderTypes.BINARY:
            le = BinaryEncoder(cols=[col])
            le_name = 'binary_encoding'

        return le, le_name

    def __get_encoded_data(self, enc, data, y, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        returned encoded data

        :param enc:
        :param data:
        :param y:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return: encoded dataframe
        """
        train_enc, test_enc, data_enc = None, None, None
        if isinstance(enc, LabelEncoder):
            if data is None and y is None:
                enc.fit(X_train)
                train_enc = enc.transform(X_train)
                test_enc = None
                if X_test is not None:
                    test_enc = enc.transform(X_test)
            else:
                enc.fit(data)
                data_enc = enc.transform(data)
        else:
            if data is None and y is None:
                enc.fit(X_train, y_train)
                train_enc = enc.transform(X_train, y_train)
                test_enc = None
                if X_test is not None:
                    test_enc = enc.transform(X_test, y_test)
            else:
                enc.fit(data, y)
                data_enc = enc.transform(data, y)
        if data is None and y is None:
            return train_enc, test_enc
        else:
            return data_enc

    def __encode_by_configs(self, data=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        encode data and set the configurations
        :param data:
        :param y:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        for col in tqdm(self._cfg.ENCODER):
            encode_type = self._cfg.ENCODER[col]
            col = col.lower()
            if col == self._cfg.DATASET.TARGET:
                continue
            if X_train is not None and col not in X_train.columns:
                continue
            if data is not None and col not in data.columns:
                continue
            enc, enc_name = self.__get_encoder(encoder_type=encode_type, col=col)
            if encode_type == EncoderTypes.LABEL:
                if data is None and y is None:
                    train_val = X_train[col].values
                    test_val = X_test[col].values

                    X_train[col], X_test[col] = self.__get_encoded_data(enc=enc, data=None, y=None, X_train=train_val,
                                                                        X_test=test_val)
                else:
                    train_val = data[col].values
                    data[col] = self.__get_encoded_data(enc=enc, data=train_val, y=y)
            else:
                if data is None and y is None:
                    X_train, X_test = self.__get_encoded_data(enc=enc, data=None, y=None, X_train=X_train,
                                                              X_test=X_test,
                                                              y_train=y_train, y_test=y_test)
                else:
                    data = self.__get_encoded_data(enc=enc, data=data, y=y)

        if data is None and y is None:
            return X_train, X_test
        else:
            return data

    def do_encode(self, data=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        apply encoders
        :param data:
        :param y:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        params = {
            'data': data,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return self.__encode_by_configs(**params)

    def custom_encoding(self, data, col, encode_type):
        """
        if you want to use custom encoders for a specific column

        :param data:
        :param col:
        :param encode_type:
        :return:
        """
        enc, enc_name = self.__get_encoder(encoder_type=encode_type, col=col)
        return enc.fit_transform(data[col])
