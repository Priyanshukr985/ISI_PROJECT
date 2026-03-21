import os
import tempfile

from werkzeug.utils import secure_filename


class ImageService:
    """Extracts text from uploaded question images using lightweight OCR when available."""

    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self):
        self._pytesseract = None
        self._pil_image = None
        self._load_ocr_dependencies()

    def _load_ocr_dependencies(self):
        try:
            import pytesseract
            from PIL import Image

            self._pytesseract = pytesseract
            self._pil_image = Image
        except ImportError:
            self._pytesseract = None
            self._pil_image = None

    def allowed_file(self, filename):
        extension = os.path.splitext(filename or "")[1].lower()
        return extension in self.ALLOWED_EXTENSIONS

    def extract_text(self, uploaded_file):
        filename = secure_filename(uploaded_file.filename or "")
        if not filename or not self.allowed_file(filename):
            raise ValueError("Please upload a PNG or JPG image.")

        if self._pytesseract is None or self._pil_image is None:
            raise RuntimeError(
                "Image OCR is not available yet. Install Pillow and pytesseract to enable image solving."
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                uploaded_file.save(temp_file.name)
                temp_path = temp_file.name

            image = self._pil_image.open(temp_path)
            text = self._pytesseract.image_to_string(image)
            cleaned = " ".join(text.split())
            if not cleaned:
                raise ValueError("I could not read clear text from that image.")
            return cleaned
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
