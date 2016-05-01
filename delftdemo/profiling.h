//
//  profiling.h
//  delftdemo
//
//  Created by Kevin Dunn on 2016-04-21.
//  Copyright © 2016 Kevin Dunn. All rights reserved.
//

#ifndef profiling_h
#define profiling_h

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;

class Profile{
private:
    vector<chrono::milliseconds::rep> times;
    vector<string> step_name_;
    int buffer_idx = 0;
public:
    std::chrono::high_resolution_clock::time_point begin;
    
    
    // Three different constructor options
    Profile();
    Profile(const Profile &);   // Copy constructor
    ~Profile();
    
    // Public member functions
    inline void start() {
        this->begin = chrono::high_resolution_clock::now();
        buffer_idx = 0;
    }
    inline void next(const string step_name) {
        auto end = chrono::high_resolution_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();;
        cout << "[" << step_name << "] took: " << elapsed_time << endl;
        times.push_back(elapsed_time);
        step_name_.push_back(step_name);
        begin = end;
        buffer_idx++;
    }
    inline void loop_back(const string step_name="") {
        this->next(step_name); // close off the profiler, and return it back to the start
        this->start(); }
    inline void stop() { this->next("STOP"); }
};

#endif /* profiling_h */

// Class constructor
Profile::Profile(){
    this->start();
}
Profile::Profile(const Profile &){
    assert("Do not copy a profile class object.");
}
Profile::~Profile(){
    this->stop();
}