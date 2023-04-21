#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
   if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
      return 1;
   }

   std::ifstream infile(argv[1]);
   if (!infile) {
      std::cerr << "Error: could not open file " << argv[1] << std::endl;
      return 1;
   }

   std::string hash_mark;
   int n, m, v1, v2;
   infile >> hash_mark >> n >> m;  // extract integers from first line
   infile >> v1 >> v2;

   std::cout << "n = " << n << std::endl;
   std::cout << "m = " << m << std::endl;

   std::cout << "v1 = " << v1 << std::endl;
   std::cout << "v2 = " << v2 << std::endl;

   return 0;
}
