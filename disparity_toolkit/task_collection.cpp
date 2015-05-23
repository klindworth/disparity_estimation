#include "task_collection.h"

#include <boost/filesystem.hpp>
#include <iostream>

TaskTestSet::TaskTestSet(const std::string& filename) : task_collection(filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(!stream.isOpened())
		std::cerr << "failed to open " << filename << std::endl;

	std::vector<std::string> taskFilenames;
	stream["tasks"] >> taskFilenames;

	for(const std::string& cname : taskFilenames)
		tasks.push_back(stereo_task::load_from_file(cname));
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskTestSet& testset)
{
	stream << "tasks" << "[:";
	for(const stereo_task& ctask : testset.tasks)
		stream << ctask.name;
	stream << "]";

	return stream;
}

std::vector<std::string> get_filenames(const std::string& folder_name, const std::string& file_extension)
{
	std::vector<std::string> result;

	boost::filesystem::path cpath(folder_name);
	try
	{
		if (boost::filesystem::exists(cpath))
		{
			if (boost::filesystem::is_directory(cpath))
			{
				auto it = boost::filesystem::directory_iterator(cpath);
				auto end = boost::filesystem::directory_iterator();

				for(; it != end; ++it)
				{
					boost::filesystem::path cpath = *it;
					if(boost::filesystem::is_regular_file(cpath) && cpath.extension() == file_extension)
						result.push_back(cpath.filename().string());
				}
			}
		}
		else
			std::cerr << folder_name + " doesn't exist" << std::endl;
	}
	catch (const boost::filesystem::filesystem_error& ex)
	{
		std::cout << ex.what() << std::endl;
	}

	return result;
}

std::string build_filename(const boost::filesystem::path& fpath, const std::string& filename)
{
	boost::filesystem::path cpath = fpath;
	cpath /= filename;

	return cpath.string();
}

bool check_completeness(const std::vector<std::string>& filenames, const boost::filesystem::path& fpath)
{
	for(const std::string& cfilename : filenames)
	{
		boost::filesystem::path cpath = fpath;
		cpath /= cfilename;

		if(!boost::filesystem::exists(cpath))
			return false;
	}

	return true;
}

void enforce_completeness(std::vector<std::string>& filenames, const boost::filesystem::path& fpath)
{
	filenames.erase(std::remove_if(filenames.begin(), filenames.end(), [=](const std::string& cfilename) {
		boost::filesystem::path cpath = fpath;
		cpath /= cfilename;

		return !boost::filesystem::exists(cpath);
	}), filenames.end());
}

folder_testset::folder_testset(const std::string& filename) : task_collection(filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(!stream.isOpened())
		std::cerr << "failed to open " << filename << std::endl;

	//file stuff
	stream["left"] >> left;
	stream["right"] >> right;
	stream["groundLeft"] >> dispLeft;
	stream["groundRight"] >> dispRight;
	stream["fileextension_images"] >> fileextension_images;
	stream["fileextension_gt"] >> fileextension_gt;

	std::vector<std::string> filenames = get_filenames(left, fileextension_images);

	bool require_completeness = true;
	if(require_completeness)
	{
		check_completeness(filenames, right);
		if(!dispLeft.empty())
			enforce_completeness(filenames, dispLeft);
		if(!dispRight.empty())
			enforce_completeness(filenames, dispRight);
	}
	else
	{
		if(! check_completeness(filenames, right))
			std::cerr << "right folder not complete" << std::endl;
		if(!dispLeft.empty())
		{
			if(!check_completeness(filenames, dispLeft))
				std::cerr << "dispLeft folder not complete" << std::endl;
		}
		if(!dispRight.empty())
		{
			if(!check_completeness(filenames, dispRight))
				std::cerr << "dispRight folder not complete" << std::endl;
		}
	}

	std::cout << filenames.size() << " files found" << std::endl;

	//settings
	stream["dispRange"] >> dispRange;
	stream["groundTruthSubsampling"] >> subsamplingGroundTruth;

	//create tasks
	for(const std::string& cfilename : filenames)
	{
		std::string file_left = build_filename(left, cfilename);
		std::string file_right = build_filename(right, cfilename);
		std::string file_dispLeft = !dispLeft.empty() ? build_filename(dispLeft, cfilename) : "";
		std::string file_dispRight = !dispRight.empty() ? build_filename(dispRight, cfilename) : "";

		std::cout << cfilename << std::endl;

		tasks.emplace_back(cfilename, file_left, file_right, file_dispLeft, file_dispRight, "", "", subsamplingGroundTruth, dispRange);
	}

	std::cout << tasks.size() << " tasks found" << std::endl;
}
