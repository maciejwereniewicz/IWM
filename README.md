# Symulator Tomografii Komputerowej

## 1. Skład zespołu

- Mateusz Graja, 155901
- Maciej Wereniewicz, 155915

## 2. Zastosowany model tomografu

W projekcie zastosowano model równoległy. W tym modelu emiter i detektory nie obracają obrazu, lecz obracają się względem statycznego obrazu. Obliczenia wartości dla poszczególnych detektorów realizowane są poprzez wyznaczenie linii (wierszy) przechodzących przez obraz – stosowany jest algorytm Bresenhama.

## 3. Zastosowany język programowania oraz biblioteki

Projekt został zaimplementowany w języku Python z wykorzystaniem bibliotek:

- **NumPy** – do obliczeń numerycznych i operacji na macierzach.
- **OpenCV** – do wczytywania oraz przetwarzania obrazów.
- **Matplotlib** – do wizualizacji wyników (wykresy i obrazy).
- **ipywidgets** – do budowania interfejsu interaktywnego w notatniku Jupyter.
- **pydicom** – do odczytu i zapisu plików DICOM.

## 4. Opis głównych funkcji programu

### 4.1 Pozyskiwanie odczytów z detektorów

Funkcja `compute_sinogram` w klasie `TomographySimulator` odpowiada za obliczenie sinogramu. Dla każdego z zadanych kątów oraz dla każdej pozycji detektora obliczana jest średnia wartość pikseli na dyskretnie wyznaczonej linii. Poniższy fragment kodu ilustruje tę operację:

```python
def compute_sinogram(self):
    n_angles = len(self.angles_rad)
    sinogram = np.zeros((n_angles, self.n_detectors))
    rays = []
    for i, theta in enumerate(self.angles_rad):
        rays_angle = []
        for j, offset in enumerate(self.detector_offsets):
            start_x = self.cx + offset * np.cos(theta + np.pi/2) - self.R * np.cos(theta)
            start_y = self.cy + offset * np.sin(theta + np.pi/2) - self.R * np.sin(theta)
            end_x   = self.cx + offset * np.cos(theta + np.pi/2) + self.R * np.cos(theta)
            end_y   = self.cy + offset * np.sin(theta + np.pi/2) + self.R * np.sin(theta)
            line_pixels = bresenham_line(start_x, start_y, end_x, end_y)
            valid_pixels = [(x, y) for (x, y) in line_pixels if 0 <= x < self.width and 0 <= y < self.height]
            if valid_pixels:
                intensities = [self.image[y, x] for (x, y) in valid_pixels]
                value = np.mean(intensities)
            else:
                value = 0
            sinogram[i, j] = value
            rays_angle.append(valid_pixels)
        rays.append(rays_angle)
    self.sinogram = sinogram
    self.rays = rays
    return sinogram
```

### 4.2 Filtrowanie sinogramu

Do filtrowania zastosowano filtr rampowy (splot w dziedzinie częstotliwości) dla redukcji szumu wynikającego z dyskretnej, skończonej odwrotnej transformacji. W naszej implementacji zastosowaliśmy filtr rampowy (operujący w dziedzinie częstotliwości) jako opcję, którą użytkownik może włączyć lub wyłączyć. Funkcja `apply_ramp_filter` wygląda następująco:

```python
def apply_ramp_filter(self):
    if self.sinogram is None:
        raise ValueError("Najpierw oblicz sinogram!")
    filtered = np.zeros_like(self.sinogram)
    for i in range(self.sinogram.shape[0]):
        projection = self.sinogram[i, :]
        proj_fft = np.fft.fft(projection)
        freqs = np.fft.fftfreq(self.n_detectors, d=1.0)
        ramp = np.abs(freqs)
        proj_fft_filtered = proj_fft * ramp
        proj_filtered = np.fft.ifft(proj_fft_filtered).real
        filtered[i, :] = proj_filtered
    self.filtered_sinogram = filtered
    return filtered
```

### 4.3 Rekonstrukcja obrazu

Rekonstrukcja odbywa się poprzez prostą metodę filtrowanego odwrotnego rzutowania. Każdej projekcji przypisywany jest wkład do pikseli obrazu wynikowego, sumowany z uwzględnieniem odpowiedniego współczynnika. Dodatkowo, przed zapisem do pliku DICOM, obraz jest normalizowany do zakresu 0–255.

```python
def reconstruct(self):
    if self.filtered_sinogram is None:
        self.apply_ramp_filter()
    reconstruction = np.zeros((self.height, self.width))
    self.bp_angles_contrib = []
    for i, theta in enumerate(self.angles_rad):
        contrib = np.zeros((self.height, self.width))
        for j, offset in enumerate(self.detector_offsets):
            p_val = self.filtered_sinogram[i, j]
            ray = self.rays[i][j]
            for (x, y) in ray:
                contrib[y, x] += p_val
        reconstruction += contrib
        self.bp_angles_contrib.append(contrib)
    reconstruction = reconstruction * (np.pi / len(self.angles_rad))
    self.reconstruction = reconstruction
    return reconstruction
```

### 4.4 Wyznaczanie miary RMSE

W ramach analizy jakości rekonstrukcji obliczana jest miara RMSE (Root Mean Square Error) – średni błąd między oryginalnym obrazem a zrekonstruowanym. RMSE obliczany jest jako pierwiastek średniej z kwadratów różnic pikseli. W interfejsie udostępniony jest przycisk, który generuje wykres RMSE w zależności od liczby wykorzystanych kątów podczas iteracyjnej rekonstrukcji.

### 4.5 Odczyt i zapis plików DICOM

Do obsługi plików DICOM wykorzystano bibliotekę `pydicom`. Program umożliwia zarówno odczyt – przy wgrywaniu pliku DICOM, jak i zapis wyniku rekonstrukcji do pliku DICOM. Przykładowy fragment kodu zapisu wygląda następująco:

```python
def save_dicom(b):
    if simulator is None or simulator.reconstruction is None:
        with output_area:
            print("Brak obrazu do zapisu. Najpierw uruchom symulację.")
        return

    recon = simulator.reconstruction
    final_image = convert_image_to_ubyte(recon)
    
    patient_name = patient_name_text.value
    patient_id = patient_id_text.value
    study_date = date_picker.value or datetime.date.today()
    study_date_str = study_date.strftime("%Y%m%d")
    filename = dicom_filename_text.value.strip() or "output.dcm"
    
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    dt = datetime.datetime.now()
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "T"  
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDate = study_date_str
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S")
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.Rows, ds.Columns = final_image.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    ds.WindowCenter = int(np.median(final_image))
    ds.WindowWidth = int(np.max(final_image) - np.min(final_image))

    ds.PixelData = final_image.tobytes()

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.Manufacturer = "MyTomographySimulator"
    ds.StudyDescription = comment_text.value

    try:
        ds.save_as(filename, write_like_original=False)
        with output_area:
            print(f"Pomyślnie zapisano DICOM jako: {filename}")
    except Exception as e:
        with output_area:
            print(f"Błąd zapisu DICOM: {str(e)}")
```

## 5. Przykład działania programu

Program umożliwia wczytanie obrazu wejściowego w formacie JPG lub DICOM oraz wykonanie pełnej symulacji tomografii. Wynik prezentowany jest w trzech panelach:

- Obraz oryginalny
- Sinogram
- Obraz zrekonstruowany

Opcjonalna interaktywna rekonstrukcja iteracyjna oraz filtrowanie sinogramu.

## 6. Eksperyment – wpływ parametrów na jakość obrazu (RMSE)

Przeprowadzono analizę wpływu parametrów na RMSE:

- Liczba detektorów: 90–720
- Liczba skanów: 90–720
- Rozpiętość wachlarza: 45°–270°

Wnioski:

- Więcej detektorów i skanów redukuje RMSE.
- Większa rozpiętość wachlarza poprawia jakość rekonstrukcji.
- Filtracja sinogramu znacząco zmniejsza RMSE i poprawia jakość rekonstrukcji.

## 7. Podsumowanie

W projekcie zaimplementowano symulator tomografii komputerowej został zaimplementowany w języku Python z zastosowaniem interfejsu opartego na ipywidgets. System wykorzystuje model równoległy, umożliwia wczytywanie obrazów zarówno w formacie JPG, jak i DICOM oraz zapis wyników w standardzie DICOM. Dodatkowo aplikacja udostępnia opcjonalne filtrowanie sinogramu (filtr rampowy) oraz analizę statystyczną jakości rekonstrukcji za pomocą miary RMSE. Przeprowadzone eksperymenty jednoznacznie wskazują, że zwiększenie liczby detektorów i skanów poprawia jakość odwzorowania, a zastosowanie filtracji istotnie redukuje szumy.
