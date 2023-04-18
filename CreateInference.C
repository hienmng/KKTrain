/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro parses a Keras .h5 file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta

using namespace TMVA::Experimental;

// bname = base name of hdf5 model file name created by keras.save()
// suffix = suffix on base name
// code will be put in the code directory using just the base name
void CreateInference(const char* bname,const char* suffix=""){
  string modelname = string("models/") + string(bname)+ string(suffix) + string(".h5");
  string codename = string("code/") + string(bname) + string(".hxx");
  cout << "Parsing file " << modelname << endl;
  //Parsing the saved Keras .h5 file into RModel object
  SOFIE::RModel model = SOFIE::PyKeras::Parse(modelname);

  //Generating inference code
  model.Generate();
  cout << "Saving file " << codename << endl;
  model.OutputGenerated(codename);
}
