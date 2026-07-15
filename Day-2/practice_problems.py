#Lists, Sets, Tuples, Dictionaries
# Problem 1: Find the largest number in a list
def find_largest(numbers):
    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    return largest

print(find_largest([3, 7, 2, 9, 4]))

# Problem 2: Find the second largest number in a list
def find_second_largest(numbers):
    largest = float('-inf')
    second_largest = float('-inf')
    
    for num in numbers:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest:
            second_largest = num
    
    return second_largest

print(find_second_largest([3, 7, 2, 9, 4]))

#Problem 3: Remove duplicate values from a list
def remove_duplicates(numbers):
    return list(set(numbers))

print(remove_duplicates([1, 2, 2, 3, 4, 4, 5]))

#Problem 4: Reverse a list without using built-in reverse().
def reverse_list(numbers):
    reversed_list = []
    for i in range(len(numbers) - 1, -1, -1):
        reversed_list.append(numbers[i])
    return reversed_list

print(reverse_list([1, 2, 3, 4, 5]))

#Problem 5: Find common elements between two lists.
def common_elements(list1, list2):
    common = []
    for item in list1:
        if item in list2:
            common.append(item)
    return common

print(common_elements([1, 2, 3, 4], [3, 4, 5, 6]))

#Problem 6: Count occurrences of an element.
def count_occurrences(data, target):
    count = 0
    for item in data:
        if item == target:
            count += 1
    return count

print(count_occurrences((1, 2, 3, 2, 4, 2), 2))

#Problem 7: Convert a tuple into a list.
my_tuple = (1, 2, 3)
my_list = list(my_tuple)

my_list2 = [4, 5, 6]
my_tuple2 = tuple(my_list2)

print(my_list)
print(my_tuple2)

#Problem 8: Find unique values from a list.
def unique_values(numbers):
    return list(set(numbers))

print(unique_values([1, 2, 2, 3, 4, 4, 5]))

#Problem 8: Perform union and intersection operations.
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union_result = set1 | set2          #  set1.union(set2)
intersection_result = set1 & set2   # set1.intersection(set2)

print(union_result)
print(intersection_result)

#Problem 9: Create a student record dictionary.
students = {
    "student1": {"name": "Ali", "age": 20, "marks": 85},
    "student2": {"name": "Hifsa", "age": 21, "marks": 80},
    "student3": {"name": "Sara", "age": 22, "marks": 90}
}

print(students)

#Problem 10: Calculate average marks of students.
def average_marks(students):
    total = 0
    count = 0
    for student_id, details in students.items():
        total += details["marks"]
        count += 1
    return total / count

print(average_marks(students))

# Problem 11: Count frequency of words in a sentence.
def word_frequency(sentence):
    words = sentence.split()
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency

print(word_frequency("the quick brown fox jumps over the lazy dog the fox runs"))