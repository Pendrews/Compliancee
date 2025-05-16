from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import sys
import pandas as pd
import smtplib
from email.message import EmailMessage
import os

# Email Credentials (Replace with your credentials)
EMAIL_ADDRESS = 'your_email@example.com'
EMAIL_PASSWORD = 'your_email_password'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587


class ContactSenderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Email Sender")
        self.setGeometry(100, 100, 400, 200)

        self.excel_file_path = ""
        self.pdf_file_path = ""

        layout = QtWidgets.QVBoxLayout()

        self.upload_excel_btn = QtWidgets.QPushButton("Upload Excel")
        self.upload_excel_btn.clicked.connect(self.upload_excel)
        layout.addWidget(self.upload_excel_btn)

        self.upload_pdf_btn = QtWidgets.QPushButton("Attach PDF")
        self.upload_pdf_btn.clicked.connect(self.upload_pdf)
        layout.addWidget(self.upload_pdf_btn)

        self.send_btn = QtWidgets.QPushButton("Send Emails")
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self.send_messages)
        layout.addWidget(self.send_btn)

        self.setLayout(layout)

    def upload_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self.excel_file_path = file_path
            QMessageBox.information(self, "Success", "Excel file uploaded successfully!")
            self.check_ready()

    def upload_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_file_path = file_path
            QMessageBox.information(self, "Success", "PDF file attached successfully!")
            self.check_ready()

    def check_ready(self):
        if self.excel_file_path and self.pdf_file_path:
            self.send_btn.setEnabled(True)

    def send_messages(self):
        try:
            df = pd.read_excel(self.excel_file_path)

            if 'Email' not in df.columns:
                QMessageBox.critical(self, "Error", "Excel file must contain an 'Email' column.")
                return

            for index, row in df.iterrows():
                email = row['Email']

                if pd.notna(email):
                    self.send_email(email)

            QMessageBox.information(self, "Success", "Emails sent successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def send_email(self, recipient_email):
        msg = EmailMessage()
        msg['Subject'] = "Your Attachment"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg.set_content("Please find the attached document.")

        with open(self.pdf_file_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=os.path.basename(self.pdf_file_path))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContactSenderApp()
    window.show()
    sys.exit(app.exec_())
