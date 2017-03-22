// Copyright (c) 2017 Evan S Weinberg
// Header file for a vector storage object which essentially performs
// memory management for arrays of a fixed datatype and a fixed length.
// The user creates an ArrayStorageMG object and specifies:
// * The datatype of the arrays that should be created.
// * The length of the arrays that should be created.
// * How many arrays to pre-allocate.
//
// The user can then "check-out" and "return" arrays. If the user
// requests a check-out but no arrays exist, a new one is created.

#ifndef QMG_ARRAY_STORAGE_OBJECT
#define QMG_ARRAY_STORAGE_OBJECT

#include <vector>

using std::vector;

// QLINALG
#include "blas/generic_vector.h"

template <typename T>
class ArrayStorageMG
{
private:
  // Get rid of copy, assignment operator.
  ArrayStorageMG(ArrayStorageMG const &);
  ArrayStorageMG& operator=(ArrayStorageMG const &);

  // Length of the stored arrays
  const int array_length;

  // Current number of allocated arrays.
  int allocated_arrays;

  // List which specifies if arrays are checked out or not.
  vector<bool> is_checked_out;
  int n_checked;

  // Vector of arrays.
  vector<T*> arrays;
  
public:
  
  // Constructor. Takes in array length and number to pre-allocate.
  ArrayStorageMG(const int length, const int n_prealloc = 1)
    : array_length(length), allocated_arrays(n_prealloc), n_checked(0)
  {
    if (n_prealloc < 1)
    {
      std::cout << "[QMG-ERROR]: ArrayStorageMG cannot preallocate less than one vector.\n";
    }

    for (int i = 0; i < n_prealloc; i++)
    {
      arrays.push_back(allocate_vector<T>(array_length));
      is_checked_out.push_back(false);
    }
  }

  // Destructor. Clean up!
  ~ArrayStorageMG()
  {
    for (int i = 0; i < allocated_arrays; i++)
      deallocate_vector(&arrays[i]);
  }

  // Check out an array. If there aren't any free arrays,
  // create a new one.
  T* check_out()
  {
    // Look for free arrays.
    for (int i = 0; i < allocated_arrays; i++)
    {
      if (!is_checked_out[i])
      {
        is_checked_out[i] = true;
        n_checked++;
        return arrays[i];
      }
    }

    // If we reach this point, we need to create a new array.
    arrays.push_back(allocate_vector<T>(array_length));
    is_checked_out.push_back(true);
    n_checked++;
    return arrays[allocated_arrays++];
  }

  // Return an array.
  void check_in(T* arr)
  {
    for (int i = 0; i < allocated_arrays; i++)
    {
      if (arrays[i] == arr)
      {
        if (is_checked_out[i])
        {
          is_checked_out[i] = false;
          n_checked--;
        }
        else
        {
          cout << "[QMG_WARNING]: Returned array that wasn't checked out.\n";
        }
        return;
      }
    }

    // If we reached here, the array didn't exist.
    cout << "[QMG_WARNING]: Returned array that doesn't live in library.\n";
  }

  // Get number allocated.
  int get_number_allocated()
  {
    return allocated_arrays;
  }

  // Get number checked out.
  int get_number_checked()
  {
    return n_checked; 
  }

  // To do: Add 'consolidate'. Free un-allocated arrays down to some minimum.
};


#endif // QMG_ARRAY_STORAGE_OBJECT
