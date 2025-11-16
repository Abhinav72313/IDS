import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

DEFAULT_NUMERICAL = [
    # frame
    'frame.number','frame.len','frame.cap_len',
    # ip / ipv6 header numeric fields (but NOT ip.src/ip.dst)
    'ip.version','ip.hdr_len','ip.len','ip.ttl','ip.frag_offset','ip.id',
    'ipv6.plen','ipv6.hlim',
    # tcp numeric
    'tcp.seq','tcp.ack','tcp.len','tcp.hdr_len','tcp.window_size_value','tcp.window_size',
    'tcp.urgent_pointer','tcp.options.mss_val','tcp.options.wscale.shift',
    'tcp.options.timestamp.tsval','tcp.options.timestamp.tsecr',
    # udp
    'udp.length',
    # tcp analysis counters (often 0/1 but numerical)
    'tcp.analysis.retransmission','tcp.analysis.fast_retransmission',
    'tcp.analysis.out_of_order','tcp.analysis.duplicate_ack',
    'tcp.analysis.keep_alive','tcp.analysis.zero_window','tcp.analysis.window_full'
]

DEFAULT_CATEGORICAL = [
    'frame.protocols',
    'eth.src','eth.dst','eth.type',
    'ip.flags','ip.flags.df','ip.flags.mf','ip.proto','ip.checksum.status',
    'ipv6.nxt',
    'tcp.srcport','tcp.dstport','tcp.flags','tcp.checksum.status',
    'udp.srcport','udp.dstport','udp.checksum.status',
    'icmp.type','icmp.code',
    'dns.flags.response','dns.flags.rcode','dns.qry.type',
    'tls.record.content_type','tls.handshake.type','tls.handshake.version'
]

def _build_vocab(series: pd.Series, top_k: int = 5000) -> Dict[str,int]:
    vals = series.fillna("<MISSING>").astype(str)
    vc = vals.value_counts()
    top = vc.head(top_k).index.tolist()
    vocab = {"<MISSING>": 0, "<UNK>": 1}
    i = 2
    for tok in top:
        if tok not in vocab:
            vocab[tok] = i
            i += 1
    return vocab

def _map_to_vocab(series: pd.Series, vocab: Dict[str,int]) -> np.ndarray:
    s = series.fillna("<MISSING>").astype(str)
    return s.map(lambda x: vocab.get(x, vocab["<UNK>"])).astype(np.int32).values

def preprocess_packets_all(
    csv_or_df,
    drop_ip_addresses: bool = True,
    top_k_vocab: int = 5000,
    drop_high_missing_frac: float = 0.99
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load (csv path or DataFrame) and preprocess numeric + categorical packet features.

    Returns:
      processed_df: pd.DataFrame (all numeric; categorical cols replaced by int tokens; numeric scaled)
      artifacts: dict containing {'scaler', 'cat_vocabs', 'num_cols', 'cat_cols', 'num_feature_names', 'cat_feature_names'}
    """
    # ---- load ----
    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(str(csv_or_df), low_memory=False)
    elif isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df.copy()
    else:
        raise ValueError("csv_or_df must be path or DataFrame")

    # normalize blanks -> NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # ---- compute delta_time and drop raw epoch ----
    if 'frame.time_epoch' in df.columns:
        # try numeric coercion
        try:
            df['frame.time_epoch'] = pd.to_numeric(df['frame.time_epoch'], errors='coerce')
            df = df.sort_values('frame.time_epoch', kind='stable').reset_index(drop=True)
            df['delta_time'] = df['frame.time_epoch'].diff().fillna(0.0).astype(float)
            # drop the raw epoch so we don't normalize absolute time later
            df = df.drop(columns=['frame.time_epoch'])
        except Exception:
            # fallback: keep delta_time = 0
            df['delta_time'] = 0.0
    else:
        df['delta_time'] = 0.0

    # ---- drop IP addresses if requested ----
    if drop_ip_addresses:
        for c in ['ip.src','ip.dst','ipv6.src','ipv6.dst']:
            if c in df.columns:
                df = df.drop(columns=c)

    # ---- select categorical and numeric columns from defaults but ensure they exist ----
    cat_cols = [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    num_cols = [c for c in DEFAULT_NUMERICAL if c in df.columns]
    # ensure delta_time included in numeric processing
    if 'delta_time' in df.columns and 'delta_time' not in num_cols:
        num_cols.insert(0, 'delta_time')

    # anything else numeric that isn't categorical and not in our lists: include it as numeric
    extra_numeric = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in num_cols and c not in cat_cols]
    for c in extra_numeric:
        # Check if column has any non-null values
        if df[c].notna().sum() > 0:
            num_cols.append(c)

    # ---- drop cols that are mostly missing or constant ----
    drop_list = []
    for c in df.columns:
        non_null_count = df[c].notna().sum()
        if non_null_count == 0 or df[c].isna().mean() >= drop_high_missing_frac:
            drop_list.append(c)
        elif df[c].nunique(dropna=True) <= 1:
            drop_list.append(c)
    
    print(f"  Dropping {len(drop_list)} columns that are mostly missing or constant")
    for c in drop_list:
        if c in cat_cols: cat_cols.remove(c)
        if c in num_cols: num_cols.remove(c)
    df = df.drop(columns=drop_list, errors='ignore')

    # ---- numeric preprocessing: missing flags, median impute, scale ----
    # create missing indicators
    num_missing_flags = []
    valid_num_cols = []  # Keep track of valid numeric columns
    
    for c in num_cols:
        # Check if column exists and has some valid data
        if c not in df.columns:
            continue
            
        # Convert column to numeric first, coercing errors to NaN
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Skip columns that are all NaN after conversion
        if df[c].notna().sum() == 0:
            print(f"  Skipping column {c} - all values are NaN")
            continue
            
        valid_num_cols.append(c)
        
        missc = f"{c}__miss"
        df[missc] = df[c].isna().astype(np.float32)
        num_missing_flags.append(missc)
        
        # median impute
        try:
            med = df[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
        except (TypeError, ValueError):
            # If median calculation fails, use 0.0 as default
            med = 0.0
        
        df[c] = df[c].fillna(med).astype(np.float32)
    
    # Update num_cols to only include valid columns
    num_cols = valid_num_cols

    # now fit scaler to numeric columns (num_cols)
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols].astype(np.float32))

    # ---- categorical preprocessing: build vocab and map tokens ----
    cat_vocabs = {}
    for c in cat_cols:
        vocab = _build_vocab(df[c], top_k=top_k_vocab)
        cat_vocabs[c] = vocab
        df[c] = _map_to_vocab(df[c], vocab)

    # ---- final dtypes ----
    for c in num_cols:
        df[c] = df[c].astype(np.float32)
    for c in num_missing_flags:
        df[c] = df[c].astype(np.float32)
    for c in cat_cols:
        df[c] = df[c].astype(np.int32)

    # ---- artifact pack ----
    artifacts = {
        "scaler": scaler,
        "cat_vocabs": cat_vocabs,
        "num_cols": num_cols,
        "num_missing_flags": num_missing_flags,
        "cat_cols": cat_cols,
        "num_feature_names": list(num_cols) + list(num_missing_flags),
        "cat_feature_names": cat_cols
    }

    return df, artifacts

def pcap_to_csv(pcap_file):
    output_file = Path(pcap_file).with_suffix(".csv")

    fields = [
        "-e", "frame.time_epoch", "-e", "frame.number", "-e", "frame.len",
        "-e", "frame.cap_len", "-e", "frame.protocols",
        "-e", "eth.src", "-e", "eth.dst", "-e", "eth.type",
        "-e", "ip.version", "-e", "ip.hdr_len", "-e", "ip.len", "-e", "ip.ttl",
        "-e", "ip.flags", "-e", "ip.flags.df", "-e", "ip.flags.mf", "-e", "ip.frag_offset",
        "-e", "ip.id", "-e", "ip.proto", "-e", "ip.checksum.status",
        "-e", "ip.src", "-e", "ip.dst",
        "-e", "ipv6.plen", "-e", "ipv6.nxt", "-e", "ipv6.hlim", "-e", "ipv6.src", "-e", "ipv6.dst",
        "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "tcp.seq", "-e", "tcp.ack",
        "-e", "tcp.len", "-e", "tcp.hdr_len", "-e", "tcp.flags",
        "-e", "tcp.window_size_value", "-e", "tcp.window_size", "-e", "tcp.checksum.status",
        "-e", "tcp.urgent_pointer", "-e", "tcp.options.mss_val", "-e", "tcp.options.wscale.shift",
        "-e", "tcp.options.sack_perm", "-e", "tcp.options.timestamp.tsval", "-e", "tcp.options.timestamp.tsecr",
        "-e", "tcp.analysis.retransmission", "-e", "tcp.analysis.fast_retransmission",
        "-e", "tcp.analysis.out_of_order", "-e", "tcp.analysis.duplicate_ack",
        "-e", "tcp.analysis.keep_alive", "-e", "tcp.analysis.zero_window", "-e", "tcp.analysis.window_full",
        "-e", "udp.srcport", "-e", "udp.dstport", "-e", "udp.length", "-e", "udp.checksum.status",
        "-e", "icmp.type", "-e", "icmp.code",
        "-e", "dns.flags.response", "-e", "dns.flags.rcode", "-e", "dns.qry.type",
        "-e", "tls.record.content_type", "-e", "tls.handshake.type", "-e", "tls.handshake.version"
    ]

    cmd = [
        "tshark", "-r", str(pcap_file),
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",
        "-Y", "(ip || ipv6 || arp)",
        "-c", "500000"
    ] + fields

    with open(output_file, "w", buffering=1) as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True, check=True)

    return str(output_file)

files = os.listdir('../pcaps')

for f in files:
    filename = f.split('.')[0]
    if f.endswith('.pcap') and files.count(f'{filename}.csv') == 0:
        pcap_to_csv(os.path.join('../pcaps',f))

csv = [f for f in os.listdir('../pcaps') if f.endswith('.csv') and f != 'packets_flat.csv']
for c in csv:
    print(f"Processing {c}...")
    try:
        processed_df, artifacts = preprocess_packets_all(os.path.join('../pcaps',c))
        
        # Save the processed dataframe
        output_name = c.replace('.csv', '_processed.csv')
        processed_df.to_csv(output_name, index=False)
        print(f"  -> Saved {output_name} with shape {processed_df.shape}")
        
    except Exception as e:
        print(f"  -> Error processing {c}: {e}")
        continue

