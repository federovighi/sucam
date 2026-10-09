# SUCAM — Installazione, dataset e training

Questa guida descrive come configurare SUCAM su Linux (x86_64) con GPU NVIDIA, preparare nuScenes, generare le label necessarie e avviare il training.

## File necessari per l'installazione

- `sucam-conda-explicit.txt`: elenco dei pacchetti Conda e delle relative build per Linux x86_64.
- `requirements-locked.txt`: dipendenze Python con versioni bloccate, escluse quelle installate separatamente (`torch`, `torchvision`, `mmcv-full`).
- `sucam-pip-freeze.txt`: elenco di riferimento dei pacchetti Python (non necessario per eseguire i comandi sotto).

I file di dipendenze devono essere disponibili nella directory da cui si lanciano i relativi comandi. Gli URL nell'elenco Conda richiedono che i pacchetti siano ancora disponibili sui repository indicati.

## 1. Creazione dell'environment Conda

Dalla cartella che contiene `sucam-conda-explicit.txt`:

```bash
conda create -n SUCAM --file sucam-conda-explicit.txt
conda activate SUCAM
```

Python 3.8.20 è già specificato nell'elenco Conda; non serve aggiungere `python=...`.

## 2. Installare PyTorch e i pacchetti Python bloccati

```bash
python -m pip install 'torch==2.1.0+cu118' 'torchvision==0.16.0+cu118' --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-locked.txt
```

Il file `requirements-locked.txt` contiene anche le dipendenze di `openmim`/`openxlab`. Non è una selezione minimale di pacchetti.

## 3. Installare MMCV

Solo dopo che PyTorch funziona:

```bash
python -m mim install 'mmcv-full==1.7.2'
```

L’installazione richiede una wheel binaria compatibile oppure la compilazione locale di MMCV.

## 4. Preparare la compilazione SUCAM

Dalla root del repository SUCAM, con l’environment `SUCAM` attivo:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export LDFLAGS="-Wl,--sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
```

La variabile `LDFLAGS` imposta il sysroot del toolchain Conda per la compilazione.

Compilare nella cartella in cui è presente setup.py:

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
```

`setup.py` imposta internamente le architetture `sm_70, sm_75, sm_80, sm_86`: `TORCH_CUDA_ARCH_LIST=8.9` non sovrascrive necessariamente questi flag. L’estensione `.so` deve essere compilata per Python 3.8 dell’environment `SUCAM`.

## 5. Preparazione del dataset nuScenes

Per generare le label necessarie a SUCAM servono **nuScenes v1.0-trainval** (metadati, immagini delle camere, point cloud LiDAR e mappe) e l'espansione **nuScenes-lidarseg v1.0**, che fornisce le annotazioni semantiche dei punti LiDAR. Quest'ultima non è inclusa nei normali archivi di nuScenes.

Scaricare il dataset principale dal [sito ufficiale nuScenes](https://www.nuscenes.org/nuscenes#download) e predisporlo in una cartella `nuscenes/`. Scaricare anche l'espansione lidarseg:

```bash
wget https://d36yt3mvayqw5m.cloudfront.net/public/nuscenes-lidarseg-v1.0/nuScenes-lidarseg-all-v1.0.tar.bz2
```

Dalla cartella che contiene sia l'archivio sia la directory `nuscenes/`, estrarre con:

```bash
tar -xvf nuScenes-lidarseg-all-v1.0.tar.bz2 -C nuscenes/
```

La struttura richiesta per `v1.0-trainval` è:

```text
nuscenes/
├── samples/
├── sweeps/
├── maps/
├── lidarseg/
│   └── v1.0-trainval/      # Annotazioni LiDAR semantiche (.bin)
├── v1.0-trainval/
│   ├── lidarseg.json       # Metadati dell'espansione lidarseg
│   └── ...
└── SUCAM_labels/           # PNG generati dallo script SUCAM
```

`SUCAM_labels/` va creata prima della generazione; si trova **nella root di nuScenes**, allo stesso livello di `samples/` e `sweeps/`. Tutti i PNG di training e validation sono memorizzati direttamente in questa cartella.

### Collegare il dataset alla repository

Dalla root della repository SUCAM, il codice si aspetta il dataset nel percorso relativo `../data/nuscenes`. Se il dataset risiede altrove, creare un symbolic link:

```bash
mkdir -p ../data
ln -s /percorso/assoluto/del/dataset/nuscenes/ ../data/nuscenes
mkdir -p ../data/nuscenes/SUCAM_labels
```

Il link `../data/nuscenes` punta alla root del dataset reale: non occorre duplicare i dati.

**Nota sulle modifiche già presenti nel fork:** `datasets/nuscenes.py` è stato adattato alla struttura standard di nuScenes: `get_nusc()` non aggiunge più `trainval` a `dataroot`; lettura (`cv2.imread`) e scrittura (`cv2.imwrite`) delle label ausiliarie utilizzano entrambe `SUCAM_labels/` al posto di `gen_labels/`. Queste modifiche sono già presenti nel repository e **non vanno applicate nuovamente**.

## 6. Generazione delle label SUCAM

Prima del training, dalla root della repository eseguire:

```bash
python gen_labels.py
```

Lo script non richiede argomenti da terminale: utilizza il dataset `trainval` collegato in `../data/nuscenes`, scorre prima il training set e poi il validation set e attiva la generazione tramite `dataset.gen_labels = True`. Per ciascuna vista camera elabora informazioni LiDAR e lidarseg, producendo un PNG con mappa di profondità e segmentazione ausiliaria. I file vengono nominati con il token nuScenes del relativo `sample_data`, ad esempio `SUCAM_labels/<camera_sample_data_token>.png`.

Le label sono richieste dal loader originale anche quando le loss ausiliarie `--seg` e `--dep` sono disabilitate. `gen_labels.py` genera i file **prima** del training; non è necessario eseguirlo a ogni avvio successivo se le label sono già complete.

## 7. Avvio del training

Dalla root della repository, con l'environment attivo:

```bash
python train.py nuscenes sucam
```

I due argomenti sono **posizionali**:

- `nuscenes` è il **nome del dataset** selezionato nel codice (`datasets['nuscenes']`); non è un percorso da scrivere nel comando. Il suo percorso reale viene costruito come `../data/nuscenes`, che può essere il symbolic link configurato sopra.
- `sucam` è il **nome dell'architettura** da usare, non il percorso di una backbone o di un checkpoint.

Per impostazione predefinita viene utilizzata la classe `vehicle`; per selezionare l'altra classe supportata si può specificare `--pos_class driveable`. Nel codice la GPU predefinita è `[0]` (prima GPU CUDA); per selezionarne un’altra usare `--gpus` seguito dagli indici.

Il training registra le metriche in TensorBoard, comprese le loss medie di training e validation per epoca, ed esegue validation e salvataggio del checkpoint al termine di ogni epoca. La directory di output predefinita è `test/` e può essere cambiata con `--logdir`.
