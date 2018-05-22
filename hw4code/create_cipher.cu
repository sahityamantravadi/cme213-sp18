#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
// You may include other thrust headers if necessary.

#include "test_macros.h"

// You will need to call these functors from
// thrust functions in the code do not create new ones

// returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool> {
    // TODO
    __host__ __device__
    bool operator() (unsigned char l) {
        return (!(l >= 'a' && l <= 'z'));
    }
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char> {
    // TODO
    __host__ __device__
    unsigned char operator() (unsigned char l) {
        if (l >= 'A' && l <= 'Z')
            return (l + 32);
        else
            return l;
    }
};

// apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int,
        unsigned char> {
    // TODO
    __host__ __device__
    apply_shift(thrust::device_ptr<unsigned int> shift_amt, int period) : shift_amt_(shift_amt),
                                                                       period_(period)
    {}

    __host__ __device__
    unsigned char operator() (const unsigned char &l, const unsigned int pos) {
        unsigned char shift_char = l + shift_amt_[pos % period_];
        return shift_char;
    }
    
    private:
        thrust::device_ptr<unsigned int> shift_amt_;
        int period_;
};

// Returns a vector with the top 5 letter frequencies in text.
std::vector<double> getLetterFrequencyCpu(
    const std::vector<unsigned char>& text) {
    std::vector<unsigned int> freq(256);

    for(unsigned int i = 0; i < text.size(); ++i) {
        freq[tolower(text[i])]++;
    }

    unsigned int sum_chars = 0;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        sum_chars += freq[c];
    }

    std::vector<double> freq_alpha_lower;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        if(freq[c] > 0) {
            freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));
        }
    }

    std::sort(freq_alpha_lower.begin(), freq_alpha_lower.end(),
              std::greater<double>());
    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));

    return freq_alpha_lower;
}

// Print the top 5 letter frequencies and them.
std::vector<double> getLetterFrequencyGpu(
    const thrust::device_vector<unsigned char>& text) {
    std::vector<double> freq_alpha_lower;
    // WARNING: make sure you handle the case of not all letters appearing
    // in the text.

    // TODO calculate letter frequency
    thrust::device_vector<unsigned char> text_chars = text;
    thrust::sort(thrust::device, text_chars.begin(), text_chars.end());

    int num_bins = thrust::inner_product(thrust::device,
                                         text_chars.begin(),
                                         text_chars.end() - 1,
                                         text_chars.begin() + 1,
                                         1,
                                         thrust::plus<int>(),
                                         thrust::not_equal_to<int>());

    thrust::device_vector<unsigned char> keys(num_bins);
    thrust::device_vector<double> values(num_bins);
    thrust::constant_iterator<double> freqs(1.0/(double)text.size());

    thrust::reduce_by_key(thrust::device,
                           text_chars.begin(),
                           text_chars.end(),
                           freqs,
                           keys.begin(),
                           values.begin());

    thrust::sort_by_key(thrust::device,
                         values.begin(),
                         values.end(),
                         keys.begin(),
                         thrust::greater<double>());

    freq_alpha_lower.resize(num_bins);
    thrust::copy(values.begin(), values.end(), freq_alpha_lower.begin());

    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));

    int top = freq_alpha_lower.size();
    std::cout << "Top " << top << " Letter Frequencies" << std::endl;
    std::cout << "-------------" << std::endl;
    for (unsigned int i = 0; i < top; ++i) printf ("%u %f\n", i, freq_alpha_lower[i]);
    
    return freq_alpha_lower;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Didn't supply plain text and period!" << std::endl;
        return 1;
    }

    std::ifstream ifs(argv[1], std::ios::binary);

    if(!ifs.good()) {
        std::cerr << "Couldn't open text file!" << std::endl;
        return 1;
    }

    unsigned int period = atoi(argv[2]);

    if(period < 4) {
        std::cerr << "Period must be at least 4!" << std::endl;
        return 1;
    }

    // load the file into text
    std::vector<unsigned char> text;

    ifs.seekg(0, std::ios::end); // seek to end of file
    int length = ifs.tellg();    // get distance from beginning
    ifs.seekg(0, std::ios::beg); // move back to beginning

    text.resize(length);
    ifs.read((char*)&text[0], length);

    ifs.close();

    thrust::device_vector<unsigned char> text_clean;
    // TODO: sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size!
    thrust::device_vector<unsigned char> text_vec = text;
    text_clean.resize(text_vec.size());

    thrust::remove_copy_if(thrust::make_transform_iterator(text_vec.begin(), upper_to_lower()),
                           thrust::make_transform_iterator(text_vec.end(), upper_to_lower()),
                           text_clean.begin(),
                           isnot_lowercase_alpha());

    int numElements = length - thrust::count_if(text_clean.begin(),
                                                text_clean.end(),
                                                isnot_lowercase_alpha());
    text_clean.resize(numElements);

    std::cout << "\nBefore ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    thrust::device_vector<unsigned int> shifts(period);
    // TODO fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption). 
    static thrust::default_random_engine rng(123);
    static thrust::uniform_int_distribution<int> gen(1,25);
    for (unsigned int i = 0; i < period; ++i) shifts[i] = gen(rng);

    std::cout << "\nEncryption key: ";

    for(int i = 0; i < period; ++i) {
        std::cout << static_cast<char>('a' + shifts[i]);
    }

    std::cout << std::endl;

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    // TODO: Apply the shifts to text_clean and place the result in
    // device_cipher_text.
    thrust::device_ptr<unsigned int> dp = &(shifts[0]);
    thrust::transform(thrust::device,
                      text_clean.begin(),
                      text_clean.end(),
                      thrust::make_counting_iterator(static_cast<int>(0)),
                      device_cipher_text.begin(),
                      apply_shift(dp, period));

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;

    std::cout << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char*)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
