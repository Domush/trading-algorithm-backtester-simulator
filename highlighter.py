from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lexer = PythonLexer()
        self.style = get_style_by_name('monokai')
        self.formats = {}

    def highlightBlock(self, text):
        for token, content in self.lexer.get_tokens(text):
            length = len(content)
            start = text.find(content)
            
            # This is a simplified highlighter. 
            # For a more robust one, we should track positions properly.
            # But for a code editor box, this often suffices if we handle it carefully.
            
            format = self._get_format(token)
            if format:
                # Find all occurrences of content in the text block
                index = text.find(content)
                while index >= 0:
                    self.setFormat(index, length, format)
                    index = text.find(content, index + length)

    def _get_format(self, token):
        if token in self.formats:
            return self.formats[token]

        q_format = QTextCharFormat()
        style_def = self.style.style_for_token(token)
        
        if style_def['color']:
            q_format.setForeground(QColor(f"#{style_def['color']}"))
        if style_def['bold']:
            q_format.setFontWeight(QFont.Bold)
        if style_def['italic']:
            q_format.setFontItalic(True)
            
        self.formats[token] = q_format
        return q_format

# Better implementation of Highlighter using pygments
class PygmentsHighlighter(QSyntaxHighlighter):
    def __init__(self, parent, style='monokai'):
        super().__init__(parent)
        self.lexer = PythonLexer()
        self.style = get_style_by_name(style)
        self.formats = {}

    def highlightBlock(self, text):
        offset = 0
        for token, value in self.lexer.get_tokens(text):
            length = len(value)
            format = self._get_format(token)
            if format:
                self.setFormat(offset, length, format)
            offset += length

    def _get_format(self, token):
        if token in self.formats:
            return self.formats[token]

        format = QTextCharFormat()
        style_def = self.style.style_for_token(token)
        
        if style_def['color']:
            format.setForeground(QColor(f"#{style_def['color']}"))
        if style_def['bold']:
            format.setFontWeight(QFont.Bold)
        if style_def['italic']:
            format.setFontItalic(True)
        if style_def['underline']:
            format.setFontUnderline(True)
            
        self.formats[token] = format
        return format
