with open('dialog_acts.dat', 'r') as file:
    lines = file.readlines()

unique_lines = list(set(lines))
with open('dialog_acts_drop_dupl.txt', 'w') as file:
    file.writelines(unique_lines)