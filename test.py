def bitonic_sort(arr, up, level=0):
    if len(arr) <= 1:
        return arr
    else:
        mid = len(arr) // 2
        first_half = bitonic_sort(arr[:mid], True, level+1)
        second_half = bitonic_sort(arr[mid:], False, level+1)
        arr = bitonic_merge(first_half + second_half, up)
        return arr

def bitonic_merge(arr, up):
    if len(arr) <= 1:
        return arr
    else:
        bitonic_compare(arr, up)
        mid = len(arr) // 2
        first_half = bitonic_merge(arr[:mid], up)
        second_half = bitonic_merge(arr[mid:], up)
        return first_half + second_half

def bitonic_compare(arr, up):
    dist = len(arr) // 2
    for i in range(dist):
        if (arr[i] > arr[i + dist]) == up:
            arr[i], arr[i + dist] = arr[i + dist], arr[i]

# Example usage
arr = [3, 7, 4, 8, 6, 2, 1, 5]
sorted_arr = bitonic_sort(arr, True)
print(sorted_arr)