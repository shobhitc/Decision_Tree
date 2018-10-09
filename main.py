import math
import sys

# This method chooses the best attribute to split the dataset on
def choose_best_attribute(data, attributes, target):
    best_attribute = attributes[0]
    max_information_gain = 0

    # Iterates through the attributes
    for attribute in attributes:

        if attribute != target:
            # Calls the information_gain method on each attribute and gets their individual information gains
            new_information_gain = information_gain(data, attributes, attribute, target)

        if new_information_gain > max_information_gain:
            max_information_gain = new_information_gain
            best_attribute = attribute

    # Returns the attribute that has the highest information gain
    return best_attribute


# Calculate information gain of an attribute
def information_gain(data, attributes, attribute, target):
    frequency = {}
    subset_entropy = 0.0

    # Get the index of the attribute to split on
    index = attributes.index(attribute)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if entry[index] in frequency:
            frequency[entry[index]] += 1
        else:
            frequency[entry[index]] = 1


    for key in frequency.keys():

        probability = frequency[key] / sum(frequency.values())

        data_subset = []
        for entry in data:
            if entry[index] == key:
                data_subset.append(entry)

        entropy = calculate_entropy(data_subset, attributes, target)
        subset_entropy += probability * entropy

    # Subtract entropy of the subset from the entropy of the whole set to get the info gain
    information_gain = calculate_entropy(data, attributes, target) - subset_entropy

    return information_gain


# Calculates entropy of data
def calculate_entropy(data, attributes, target):
    target_frequency = {}
    entropy = 0.0

    # Find index of the target attribute
    index = attributes.index(target)

    # Find the frequency of each target value
    for entry in data:
        if entry[index] in target_frequency:
            target_frequency[entry[index]] += 1
        else:
            target_frequency[entry[index]] = 1

    total_data_entries = sum(target_frequency.values())
    for key in target_frequency:
        key_probability = target_frequency[key] / total_data_entries
        entropy -= key_probability * (math.log(key_probability) / math.log(4))

    return entropy


# This method returns a list of unique values in the chosen attribute field
def get_values(data, attributes, attribute):
    # Get the index of the chosen attribute
    index = attributes.index(attribute)

    values = []

    # Check each row, if the value isn't already in the values list, add it to the list
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values


# Gets tuples for a particular attribute value (Ex: all tuples where safety is low)
def get_data(data, attributes, best_attribute, value):
    examples = [[]]
    index = attributes.index(best_attribute)

    for entry in data:
        if entry[index] == value:
            new_entry = []
            # Add value if it is not in best column
            for i in range(0, len(entry)):
                if i != index:
                    new_entry.append(entry[i])
            examples.append(new_entry)

    examples.remove([])
    return examples


# Get all values of the target attribute (Ex: unacc, unacc, unacc, acc, acc, good)
def get_target_values(data, attributes, target):
    values = []

    for record in data:
        index_of_target = attributes.index(target)
        value = record[index_of_target]
        values.append(value)

    return values


# Get the majority classification
def get_majority(data, attributes, target):
    frequency ={}
    index = attributes.index(target)
    for tuple in data:
        if (frequency.has_key(tuple[index])):
            frequency[tuple[index]] += 1
        else:
            frequency[tuple[index]] = 1
    max = 0
    major = ""
    for key in frequency.keys():
        if frequency[key] > max:
            max = frequency[key]
            major = key
    return major


xml_string = ""
def build_tree(data, attributes, target, first_run):
    global xml_string

    # First call
    if first_run:
        # XML header for first run
        with open('output_final.xml', 'w') as append_file:
            append_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>")
        xml_string = ""

        index = attributes.index(target)
        val_freq = {}
        # Calculate the frequency of each of the values in the target attribute
        for entry in data:
            if entry[index] in val_freq:
                val_freq[entry[index]] += 1
            else:
                val_freq[entry[index]] = 1

        # Gets entropy of the entire dataset
        class_entropy = calculate_entropy(data, attributes, target)

        val_freq_str = ""
        for val in val_freq:
            val_freq_str = val_freq_str + str(val) + ":" + str(val_freq[val]) + ","
        class_entropy = str(class_entropy)

        new_xml ="<tree classes =\"" + val_freq_str + "\" entropy=\"" + class_entropy+"\">"
        xml_string = xml_string + new_xml

        with open('output_final.xml', 'a') as file:
            file.write(xml_string)
        xml_string = ""

    # Get all values of the target attribute (Ex: unacc, unacc, unacc, acc, acc, good)
    values = get_target_values(data, attributes, target)

    # If all records in the data list have the same classification, return that classification.
    if values.count(values[0]) == len(values):
        return values[0]

    # If the dataset is empty or the attributes list is empty, return the most frequent classification
    elif not data or (len(attributes) - 1) <= 0:
        return get_majority(data, attributes, target)

    else:

        first_run = False
        # Choose the next best attribute for classification
        best_attribute = choose_best_attribute(data, attributes, target)

        # Values holds the unique values for the best attribute field. (ex: doors: 2, 4, more)
        values = get_values(data, attributes, best_attribute)
        for value in values:

            data_for_subtree = get_data(data, attributes, best_attribute, value)

            new_attributes = attributes[:]
            new_attributes.remove(best_attribute)

            if not first_run:
                index = new_attributes.index(target)
                val_freq = {}
                # Calculate the frequency of each of the values in the target attribute
                for entry in data_for_subtree:
                    if entry[index] in val_freq:
                        val_freq[entry[index]] += 1
                    else:
                        val_freq[entry[index]] = 1

                class_entropy = calculate_entropy(data_for_subtree, new_attributes, target)

                val_freq_str = ''
                for each in val_freq:
                    val_freq_str = val_freq_str + str(each) + ":" + str(val_freq[each]) + ","

                if class_entropy <= 0.0:
                    temp_str = best_attribute + "=\"" + value + "\">" + each
                else:
                    temp_str = best_attribute + "=\"" + value + "\">"

                class_entropy = str(class_entropy)

                xml_string = "<node classes=\"" + val_freq_str + "\" entropy=\"" + class_entropy + "\" " + temp_str

                with open('output_final.xml', 'a') as append_file:
                    append_file.write(xml_string)

                xml_string = ''

            build_tree(data_for_subtree, new_attributes, target, first_run)

            with open('output_final.xml', 'a') as append_file:
                append_file.write("</node>")


def main():
    # List to store the dataset
    data = []

    print(sys.executable)

    # Read from file and store in data list
    with open('car.data', 'r') as file:
        for line in file:
            line = line.strip("\r\n")
            data.append(line.split(','))

    # List of all the attributes
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
    target = 'acceptability'

    first_run = True
    build_tree(data, attributes, target, first_run)

    # Append XML end tree tag
    with open('output_final.xml', 'a') as append_file:
        append_file.write("</tree>")

    print('End')

main()
