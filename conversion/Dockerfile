# Use a lightweight Alpine base
FROM alpine:3.20

# Install LibreOffice, Java (required for certain LO plugins/features), and essential fonts
RUN apk update && apk add --no-cache \
    libreoffice \
    openjdk8-jre-base \
    # Essential fonts to prevent layout issues and missing characters
    ttf-dejavu \
    ttf-droid \
    ttf-freefont \
    ttf-liberation \
    fontconfig \
    && fc-cache -f

# Create working directory
WORKDIR /app

COPY find_doc.sh find_docx.sh find_ppt.sh find_pptx.sh convert_to_pdf.sh /app

# Default command shows LibreOffice version to verify installation
CMD ["libreoffice", "--version"]
