# Create a script that analyzes a DNA sequence.

# Write a function that Queries the user for a DNA sequence
def query_user():
    out = input('Input your DNA sequence here: ')
    return out
DNA = query_user()

# Use error checking to determine if the user only entered A,C,G,Ts (uppercase or lowercase). Return True if the
# sequence is valid, else return False
def check_sequence(seq):
    seq = str(seq)
    acceptable = ['a','c','t','g','A','C','G','T']
    count = 0
    for c in seq:
        if c in acceptable:
            count += 1
        else: 
            count += 0   
    if count == len(seq):
        return True
    else:
        return False
        

x = check_sequence(DNA)
print(x)


# Create a function that prints out the GC content (i.e., the percentage of nucleotides that are G or C) in a given
# sequence
def print_GC_content(seq):
    pass
    # your code here
    print('GC-content: {}'.format(gc_component))
    # note that .format is a string method you may not have seen before. It allows you to define a string containing
    # the special characters {} (indicating a replacement field), which is replaced by the argument passed to .format.
    # this is a very powerful string method that can replace many of the clunkier (albeit simpler) methods we have
    # seen so far.

# Create a function that prints out the reverse complement of the DNA sequence
def print_reverse_compliment(seq):
    pass
    # your code here
    print('Reverse Complimennt: {}'.format(reverse_compliment))

# Create a function that prints out the first X nucleotides starting from the Y position (e.g. 5 nucletides
# starting from the 3rd nucleotide). X and Y are arguments of the function.
def print_segment(X, Y):
    pass
    # your code here
    print('{} nucleotides starting from nucleotide {}: {}'.format(X, Y, segment))
    # note here that we are using .format with multiple replacement fields and multiple arguments. The first argument
    # will fill the first field, the second argument the second field, and so on.

# Write a function that queries the user for a DNA sequence, checks it for errors, prints the GC compliment, and prints
# the reverse compliment. You should be able to accomplish this using the functions you have already written
def analyze_sequence():
    pass
    # your code here
