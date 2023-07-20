
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


label = ["Astro","GBM","Oligo"]

pred = ["Astro","Astro","Astro","Astro","GBM","GBM","GBM","GBM","GBM","GBM","GBM","GBM","GBM","Oligo","Oligo","Oligo","GBM","GBM","Astro"]

gt =   ["Astro","Astro","Astro","Astro","Astro","Astro","GBM","GBM","GBM","GBM","GBM","GBM","GBM","Oligo","Oligo","Oligo","Oligo","Oligo","Oligo"]


cm = confusion_matrix(gt, pred, labels=label)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=label)
disp.plot()


plt.savefig(r"C:\Users\felix\Desktop\neuro\model_output\cf\cfFinal.jpg")



