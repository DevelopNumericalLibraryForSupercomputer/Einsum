#include <iostream>
#include <map>
#include <string>
#include <vector>

// row major
size_t unpack_array_index(const std::vector<size_t> array_index, const std::vector<size_t> tensor_shape){
    size_t unpack_index = array_index[0];
    for(size_t i=1;i<array_index.size();i++){
        unpack_index *= tensor_shape[i];
        unpack_index += array_index[i];
    }
    return unpack_index;
}

void input_parser(const std::string input_str, const std::vector<size_t> a_size, const std::vector<size_t> b_size,
                  std::string* result_index, std::vector<std::string>* inputs_exprs, std::map<std::string, size_t>* total_iter_sizes){
    //input_string에서 index 번호 추출, input tensor 2개 output tensor 1개
    //std::string input_str(input_string); // 문자열 리터럴을 std::string으로 변환
    size_t arrow_pos = input_str.find("->");

    if (arrow_pos != std::string::npos) {
        *result_index = input_str.substr(arrow_pos + 2);
        std::string input_indices = input_str.substr(0, arrow_pos);

        size_t comma_pos = 0;
        while ((comma_pos = input_indices.find(",")) != std::string::npos) {
            inputs_exprs->push_back(input_indices.substr(0, comma_pos));
            input_indices.erase(0, comma_pos + 1);
        }
        inputs_exprs->push_back(input_indices); // 마지막 토큰 추가

    } else {
        std::cerr << "Arrow (->) not found in the input string." << std::endl;
    }

    // 결과 출력
    
    //std::cout << "Result Index: " << *result_index << std::endl;
    //std::cout << "Input Indices: ";
    //for (const std::string& index : *inputs_exprs) {
    //    std::cout << index << " ";
    //}
    //std::cout << std::endl;
    
    //각 index letter별 tensor size 저장.
    //sizes stores size of each index, using key as single letter string.
    // For example, i: 5, j: 4, and so on.
    for (size_t j = 0; j < (*inputs_exprs)[0].size(); ++j) {
        std::string key(1, (*inputs_exprs)[0][j]);
        size_t size = a_size[j];
        (*total_iter_sizes)[key] = size;
    }
    for (size_t j = 0; j < (*inputs_exprs)[1].size(); ++j) {
        std::string key(1, (*inputs_exprs)[1][j]);
        size_t size = b_size[j];
        (*total_iter_sizes)[key] = size;
    }
}

std::vector<size_t> calculate_return_size(const std::string result_index, const std::map<std::string, size_t> total_iter_sizes){
    std::vector<size_t> c_size;
    int tmp_index = 0;
    for (char key : result_index) {
        std::string key_str(1, key);  // 문자를 문자열로 변환
        auto it = total_iter_sizes.find(key_str);
        //std::cout << it->first << " : " << it->second << std::endl;
        //std::cout << "C : " << tmp_index << " : ";
        if (it != total_iter_sizes.end()) {
            c_size.push_back(it->second);
            //std::cout << c_size.at(tmp_index)  << std::endl;
            tmp_index++;
        }
    }
    return c_size;
}

void einsum_core(std::string result_index, std::vector<std::string> inputs_exprs, std::map<std::string, size_t> total_iter_sizes,
                 const double* a, const double* b, double* c, const std::vector<size_t> a_size, const std::vector<size_t> b_size, std::vector<size_t> c_size){

    size_t c_total_size = 1;
    for (size_t size : c_size) {
        c_total_size *= size;
        //std::cout << size << ' ';
    }
    
    //모든 index에 대해 for 문을 돌아야함.
    //total number of iteration 계산
    size_t total_combinations = 1;

    std::map<std::string, size_t> indices;

    for (const auto& kv : total_iter_sizes) {        
        indices[kv.first] = 0;
        total_combinations *= kv.second;
        // 결과 출력
        //std::cout << "Key: " << kv.first << ", Size: " << kv.second << std::endl;
    }
    //one large loop
    for (size_t combination = 0; combination < total_combinations; ++combination) {
        //검증
        
        //for (const auto& kv : indices){
        //    std::cout << kv.first << " : " << kv.second << " ";
        //}
        //std::cout << '\t';
        
        //tensor index 확인
        std::vector<size_t> a_index;
        //std::cout << "a : (";
        for (size_t j = 0; j < inputs_exprs[0].size(); ++j) {
            std::string key(1, inputs_exprs[0][j]);
            a_index.push_back(indices[key]);
            //std::cout << key << " : " << a_index[j] << ' '; 
        }
        //std::cout << ")\t";

        std::vector<size_t> b_index;
        //std::cout << "b : (";
        for (size_t j = 0; j < inputs_exprs[1].size(); ++j) {
            std::string key(1, inputs_exprs[1][j]);
            b_index.push_back(indices[key]);
            //std::cout << key << " : " << b_index[j] << ' ';
        }
        //std::cout << ")\t";
        
        std::vector<size_t> c_index;
        //std::cout << "c : (";
        for (size_t j = 0; j < result_index.size(); ++j) {
            std::string key(1, result_index[j]);
            c_index.push_back(indices[key]);
            //std::cout << key << " : " << c_index[j] << ' ';
        }
        //std::cout << ")" << std::endl;

        c[unpack_array_index(c_index, c_size)] += a[unpack_array_index(a_index, a_size)] * b[unpack_array_index(b_index, b_size)];

        //c = a * b
        //index 변경 000, 100, 200, ..., 010, 110, ...
        for (auto it = indices.begin(); it != indices.end(); ++it){
            it->second += 1;
            if (it->second < total_iter_sizes[it->first]){
                break;
            }
            it->second = 0;
        }
    }
    return;
}

void einsum(const std::string input_str, const double* a, const double* b, double* c,
                           const std::vector<size_t> a_size, const std::vector<size_t> b_size) {

    std::string result_index;
    std::vector<std::string> inputs_exprs;
    std::map<std::string, size_t> total_iter_sizes;
    input_parser(input_str, a_size, b_size, &result_index, &inputs_exprs, &total_iter_sizes);
    // c의 tensor size 추출 
    std::vector<size_t> c_size = calculate_return_size(result_index, total_iter_sizes);
    einsum_core(result_index, inputs_exprs, total_iter_sizes, a, b, c, a_size, b_size, c_size);
    return;

}
