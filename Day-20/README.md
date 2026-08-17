```
# Day-20: OCR Document Reader

## What is OCR?
OCR (Optical Character Recognition) is a technology that reads and extracts text from images. Instead of seeing an image as pixels, OCR identifies characters, words, and sentences — the same way a human reads a document.

It is used in document scanning, invoice processing, number plate recognition, ID card reading, and many other real-world applications.

## Library Used: EasyOCR
I used EasyOCR because:
- Simple to set up and use
- Works well on printed text, signboards, and documents
- Supports 80+ languages
- No extra configuration needed compared to Tesseract

### Other Libraries Explored

| Library | Strengths | Limitations |
|---------|-----------|-------------|
| Tesseract | Open source, widely used | Needs manual configuration, poor on complex layouts |
| EasyOCR | Easy setup, good accuracy | Slower on CPU |
| PaddleOCR | Best accuracy, multilingual | Heavier to install |

## Which OCR Libraries Support Multithreading?
- **EasyOCR** — supports GPU parallelism but not native Python multithreading due to GIL
- **PaddleOCR** — supports multithreading and multiprocessing natively, making it the best choice for high-throughput pipelines
- **Tesseract** — supports multithreading via `tesseract` CLI with `--jobs` flag or through Python's `concurrent.futures`

## Preprocessing Techniques
Applied before OCR to improve accuracy:
- **Grayscale conversion** — removes color noise, simplifies the image for text detection
- **Contrast enhancement** — makes faded or low-contrast text more visible
- **Gaussian blur** — reduces noise and smooths background texture

Preprocessing helped on dark and low-quality images but sometimes reduced accuracy on already clear images — noted as a key observation.

## Project Structure
```
Day-20/
├── input/          — sample images used for OCR testing
├── output/         — extracted text files saved per image
├── ocr_practice.py — practice script testing OCR on all input images
├── app.py          — Gradio web app for interactive OCR
├── requirements.txt
└── README.md
```

## How to Run

**Practice script:**
```
python ocr_practice.py
```

**Gradio app:**
```
python app.py
```

## Challenges Faced
- Low confidence detections on complex or noisy images — filtered results below 0.3 confidence
- Preprocessing improved some images but reduced accuracy on others — no single setting works for all image types
- Handwritten text gave the lowest accuracy — EasyOCR struggles with irregular handwriting styles

## Author
Hifsa Iftikhar
```