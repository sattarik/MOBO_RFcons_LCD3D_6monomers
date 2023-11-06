import numpy as np

# Define the input lists
E1 = [0.50057461, 0.02336626, 0.01594571, 0.3903181, 0.0407821]
E2 = [0.05372573, 0.13260755, 0.23226889, 0.18924696, 0.00635455]

# Round the elements in E1 and E2 to two decimal places
E1 = [round(e1, 2) for e1 in E1]
E2 = [round(e2, 2) for e2 in E2]

# Calculate the sum of the rounded elements
sum_E1 = round(1 - sum(E1), 2)
sum_E2 = round(1 - sum(E2), 2)

# Combine the rounded elements with the sum and format as a string
formatted_E1 = str(E1 + [sum_E1])
formatted_E2 = str(E2 + [sum_E2])

# Remove square brackets and print the results
formatted_E1 = formatted_E1.replace('[', '').replace(']', '')
formatted_E2 = formatted_E2.replace('[', '').replace(']', '')

print(formatted_E1)
print(formatted_E2)

