import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Ignore warnings for cleaner output during execution
warnings.filterwarnings('ignore')

class AnomalEPreprocessor:
    """
    Handles the data preprocessing pipeline for the Anomal-E NIDS.
    Executes steps including port dropping, downsampling, target encoding, and L2 normalization.

    This mirrors Fig. 4 of the Anomal-E paper:
    Drop Port -> Downsampling -> Train/Test Split -> Feature Conversion (Target Encoding)
    -> Replace Empty/Infinite Values -> L2 Normalisation -> Train & Test Graph Generation
    (graph generation itself happens outside this class, using the final 'h' column
    produced in apply_normalization).
    """
    def __init__(self, target_cols=None):
        if target_cols is None:
            # These are the CATEGORICAL columns in the NetFlow-style dataset.
            # Everything else in the dataset (byte counts, durations, packet counts...)
            # is already numeric and is left untouched by the encoder below.
            self.target_cols = [
                'TCP_FLAGS', 'L7_PROTO', 'PROTOCOL', 'CLIENT_TCP_FLAGS', 
                'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 
                'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE'
            ]
        
        # Initialize encoders and scalers.
        # These are created ONCE here (bound to `self`) so that the SAME fitted
        # object (with parameters learned from the training set) can be reused
        # later in transform() calls on the test set, without re-fitting.
        self.encoder = ce.TargetEncoder(cols=self.target_cols)
        self.scaler = Normalizer()
        self.label_encoder = LabelEncoder()
        
    def load_and_clean_data(self, file_path, fraction=0.1, random_state=13, sanity_check=False):
        """
        Step 1 & 2: Loads NetFlow data, removes source/destination ports, 
        and applies uniform random downsampling.
        """
        # CSV support: We explicitly require the original CSV dataset from UQ 
        # (University of Queensland). We removed Parquet support because Kaggle
        # and converted Parquet versions often drop crucial columns like IP addresses,
        # which are strictly required for GNN node construction.
        #
        # sanity_check=True: lets you run the ENTIRE pipeline end-to-end on a
        # small slice (50,000 rows) in seconds, just to confirm the code runs
        # without errors, before committing to a full run that can take much
        # longer. This is purely a development/debugging aid -- it changes
        # nothing about the pipeline's logic, only how much data it loads.
        if sanity_check:
            print("[INFO] SANITY CHECK MODE: Loading only the first 50,000 rows from CSV...")
            data = pd.read_csv(file_path, nrows=50000)
        else:
            print("[INFO] Loading FULL dataset from CSV...")
            data = pd.read_csv(file_path)
            # Standardize column names: only strip hidden spaces
            data.rename(columns=lambda x: str(x).strip(), inplace=True)
        
            # [DEBUG] Print the first few columns to ensure they match our expectations
            print(f"[DEBUG] Standardized Columns: {data.columns.tolist()[:10]}")
        # Convert IP columns to string for graph construction.
        # IPs will later become node identifiers, so they must be treated as
        # categorical/string labels, not numbers to be normalised.
        # Added a safety check to crash early if the wrong dataset (without IPs) is provided.
        for col in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']:
            if col not in data.columns:
                raise KeyError(f"[CRITICAL ERROR] Column '{col}' is missing! Please make sure you are using the original UQ CSV dataset.")
            data[col] = data[col].apply(str)
            
        # Step 1: Drop Port Information to prevent superficial pattern learning.
        # Port numbers can let the model latch onto shallow shortcuts
        # (e.g. "port 4444 => attack") instead of learning genuine behavioural
        # patterns, which would hurt generalisation to unseen traffic.
        if "L4_SRC_PORT" in data.columns and "L4_DST_PORT" in data.columns:
            data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
            print("[INFO] Dropped Source and Destination Ports.")
            
        # Step 2: Uniform Random Downsampling.
        # Grouping by 'Attack' before sampling keeps the sampling STRATIFIED:
        # each attack type (and benign traffic) is downsampled independently,
        # so rare attack categories are not accidentally wiped out by a
        # purely random 10% sample of the whole dataset.
        print(f"[INFO] Downsampling data to {fraction*100}%...")
        data = data.groupby(by='Attack').sample(frac=fraction, random_state=random_state)
        
        return data
    
    def split_data(self, data, test_size=0.3, random_state=13, sanity_check=False):
        """
        Step 3: Splits the dataset into training and testing sets.
        """
        print("[INFO] Splitting data into Train and Test sets...")
        X = data.drop(columns=["Attack", "Label"])
        y = data[["Attack", "Label"]]
        
        # In sanity_check mode, the 50,000-row slice may contain an attack
        # category with only 1-2 rows, which stratified splitting cannot
        # handle (it needs at least 2 rows per class per split). Disabling
        # stratification avoids a crash during quick smoke-testing; the full
        # run (sanity_check=False) always keeps stratify=y as before.
        stratify_col = None if sanity_check else y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
            
        return X_train, X_test, y_train, y_test

    def apply_feature_conversion(self, X_train, X_test, y_train):
        """
        Step 4 & 5: Applies Target Encoding to categorical features and handles missing/infinite values.
        """
        print("[INFO] Applying Target Encoding (fitted on training data only)...")

        # TARGET ENCODING, explained with an example:
        # For a categorical column like PROTOCOL, each category (TCP, UDP, ICMP...)
        # is replaced by the MEAN of Label for all training rows that had that
        # category. E.g. if ICMP rows were attacks 100% of the time in training,
        # ICMP becomes ~1.0; if UDP was always benign, UDP becomes ~0.0.
        # This gives the model a numeric value that reflects how strongly that
        # category correlates with an attack, rather than an arbitrary integer
        # (which would wrongly imply an ordering between categories).
        #
        # Fit encoder exclusively on training data to prevent data leakage:
        # the means above are learned ONLY from X_train/y_train. When we later
        # call transform(X_test), any category value already seen in training
        # gets the SAME learned mean -- test data never influences the mean
        # itself.
        self.encoder.fit(X_train, y_train['Label'])
        
        X_train_enc = self.encoder.transform(X_train)
        X_test_enc = self.encoder.transform(X_test)
        
        # Step 5: Replace Empty and Infinity Values to 0.
        # Why these appear: a categorical value seen only in the TEST set
        # (never seen during fit) has no learned mean, so target encoding can
        # emit NaN; some internal smoothing computations can also divide by a
        # near-zero count and produce inf/-inf. Left unhandled, a single NaN
        # or inf would propagate through every matrix multiplication in
        # E-GraphSAGE and corrupt the resulting embeddings.
        print("[INFO] Replacing infinite and NaN values with 0...")
        for df in [X_train_enc, X_test_enc]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
        return X_train_enc, X_test_enc

    def apply_normalization(self, X_train, X_test):
        """
        Step 6: Applies L2 Normalization to numerical flow statistics.
        """
        print("[INFO] Applying L2 Normalization...")
        # Ignore the first two columns (IP addresses) for normalization --
        # they are node identifiers, not numeric flow statistics.
        cols_to_norm = list(set(list(X_train.columns[2:])))
        
        # NOTE on fit() here: unlike TargetEncoder or StandardScaler, sklearn's
        # Normalizer does L2 normalisation ROW-BY-ROW (each flow/sample is
        # scaled so its own L2 norm equals 1). Its fit() does not learn or
        # store any statistic from the training data -- it mainly exists to
        # satisfy the standard scikit-learn fit/transform API. Calling
        # fit_transform() separately on train and test would give an
        # identical result, since each row is normalised independently of
        # every other row.
        self.scaler.fit(X_train[cols_to_norm])
        
        X_train[cols_to_norm] = self.scaler.transform(X_train[cols_to_norm])
        X_test[cols_to_norm] = self.scaler.transform(X_test[cols_to_norm])
        
        # Concatenate normalized features into a single vector 'h' for edge features.
        # Each row's numeric columns (everything except the two IP columns) are
        # packed into one Python list per row. This 'h' column is exactly the
        # edge feature vector e_uv referred to in the Anomal-E paper -- it will
        # later be attached as the edge attribute between IPV4_SRC_ADDR and
        # IPV4_DST_ADDR when the graph is constructed.
        X_train['h'] = X_train.iloc[:, 2:].values.tolist()
        X_test['h'] = X_test.iloc[:, 2:].values.tolist()
        
        return X_train, X_test

    def encode_labels(self, train_df, test_df):
        """
        Encodes textual attack labels into integers.
        """
        # Fit on combined labels (train + test) to ensure all attack classes
        # are recognized -- unlike the encoder/scaler above, this is purely a
        # text-to-integer mapping for the FINAL evaluation label, not part of
        # the model's learning process, so fitting on the union here does not
        # introduce data leakage into training.
        self.label_encoder.fit(pd.concat([train_df["Attack"], test_df["Attack"]]))
        
        train_df["Attack"] = self.label_encoder.transform(train_df["Attack"])
        test_df["Attack"] = self.label_encoder.transform(test_df["Attack"])
        
        return train_df, test_df

    def process_pipeline(self, file_path, sanity_check=False):
        """
        Executes the entire data preprocessing pipeline.
        Returns fully preprocessed and normalized Train and Test dataframes.

        Order of operations mirrors Fig. 4 of the paper:
        1. load_and_clean_data   -> Drop Port + Downsampling
        2. split_data            -> Train/Test Split
        3. apply_feature_conversion -> Target Encoding + fill missing/infinite
        4. apply_normalization   -> L2 Normalisation + build final 'h' vector
        5. encode_labels         -> Map attack name strings to integers

        sanity_check=True runs the same logic on a small 50,000-row slice,
        for a fast end-to-end smoke test before a full run.
        """
        data = self.load_and_clean_data(file_path, sanity_check=sanity_check)
        X_train, X_test, y_train, y_test = self.split_data(data, sanity_check=sanity_check)
        
        X_train, X_test = self.apply_feature_conversion(X_train, X_test, y_train)
        X_train, X_test = self.apply_normalization(X_train, X_test)
        
        # Re-attach labels to form complete datasets for graph building
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df, test_df = self.encode_labels(train_df, test_df)
        
        print("[SUCCESS] Data Preprocessing Pipeline Completed.")
        return train_df, test_df