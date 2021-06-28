CXXFLAGS = -I include  -std=c++11 -O3 -pthread

CXX_SOURCES = simple_dilate_.cpp include/clipper/clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = simple_dilate.so

$(LIB_SO): $(CXX_SOURCES)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)