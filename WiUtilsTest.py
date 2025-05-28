
import wiutils
import pandas as pd

images = wiutils.read_images(r"C:\Users\16145\Downloads\wildlife-insights_9d52a0dc-1db5-4b7e-8713-fc0db46ff0ce_all-platform-data\wildlife-insights_9d52a0dc-1db5-4b7e-8713-fc0db46ff0ce_all-platform-data")

print(type(images))
print("         ============           ")
df = images


column_names = list(df.columns.values)
print(column_names)
