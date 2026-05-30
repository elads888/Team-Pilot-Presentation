import subprocess

def send_email(epoch, dev_loss, psnr_dev, train_loss, psnr_train, text= None):
    """Send an email using the cluster's sendmail system."""
    recipients = ["elad.sznaj@campus.technion.ac.il", "eden.strugo@campus.technion.ac.il"]
    for recipient in recipients:
        subject = f"Training Update: Epoch {epoch}"
        body = (f"Current Epoch: {epoch}\nLoss: Train Loss - {train_loss}\nTraining is in progress..."
                f"Test Loss - {dev_loss}\nTraining is in progress..."
                f"\nPSNR Train - {psnr_train} , PSNR test - {psnr_dev}")
        if text:
            body = text

        message = f"Subject: {subject}\n\n{body}"
        process = subprocess.Popen(["/usr/sbin/sendmail", recipient], stdin=subprocess.PIPE)
        process.communicate(message.encode())

        print(f"Email sent for epoch {epoch}")