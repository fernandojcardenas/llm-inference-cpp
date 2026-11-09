#include "llmi/model/safetensors_index.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace llmi;

namespace {

const char* kValid = R"({
  "metadata": {"total_size": 1234},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.norm.weight": "model-00002-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors"
  }
})";

}  // namespace

TEST(SafetensorsIndex, ParsesAValidIndex) {
  auto idx = SafetensorsIndex::parse(kValid);
  ASSERT_TRUE(idx) << idx.error();
  EXPECT_EQ(idx->weight_map().size(), 3U);
  EXPECT_EQ(idx->weight_map().at("model.embed_tokens.weight"), "model-00001-of-00002.safetensors");
  // shards() is the distinct, sorted set of filenames.
  ASSERT_EQ(idx->shards().size(), 2U);
  EXPECT_EQ(idx->shards()[0], "model-00001-of-00002.safetensors");
  EXPECT_EQ(idx->shards()[1], "model-00002-of-00002.safetensors");
}

TEST(SafetensorsIndex, MetadataIsOptionalButMustBeAnObject) {
  EXPECT_TRUE(SafetensorsIndex::parse(R"({"weight_map":{"a":"m.safetensors"}})"));
  EXPECT_TRUE(SafetensorsIndex::parse(R"({"metadata":null,"weight_map":{"a":"m.safetensors"}})"));
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"metadata":7,"weight_map":{"a":"m.safetensors"}})"));
}

TEST(SafetensorsIndex, RejectsStructuralProblems) {
  EXPECT_FALSE(SafetensorsIndex::parse("[]"));                                  // not an object
  EXPECT_FALSE(SafetensorsIndex::parse("{}"));                                  // no weight_map
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"weight_map":{}})"));                // empty weight_map
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"weight_map":[]})"));                // weight_map not an object
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"weight_map":{"a":7}})"));           // filename not a string
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"weight_map":{"a":"m.safetensors"},"extra":1})"));  // unknown key
  // Duplicate keys are rejected by the strict JSON parser.
  EXPECT_FALSE(SafetensorsIndex::parse(R"({"weight_map":{"a":"m.safetensors","a":"n.safetensors"}})"));
}

TEST(SafetensorsIndex, RejectsUnsafeShardFilenames) {
  auto with = [](const std::string& file) {
    return SafetensorsIndex::parse(R"({"weight_map":{"a":")" + file + "\"}}");
  };
  EXPECT_FALSE(with("../secrets.safetensors"));   // parent traversal
  EXPECT_FALSE(with("a/b.safetensors"));          // subdirectory
  EXPECT_FALSE(with("/etc/passwd.safetensors"));  // absolute
  EXPECT_FALSE(with("a\\\\b.safetensors"));       // backslash
  EXPECT_FALSE(with("."));                        // dot
  EXPECT_FALSE(with(".."));                       // dot-dot
  EXPECT_FALSE(with(".hidden.safetensors"));      // leading dot
  EXPECT_FALSE(with("-rf.safetensors"));          // leading dash (option-injection shape)
  EXPECT_FALSE(with("model-00001.bin"));          // wrong suffix
  EXPECT_FALSE(with("model 1.safetensors"));      // space
  EXPECT_FALSE(with("model$1.safetensors"));      // other punctuation
  EXPECT_FALSE(with(""));                          // empty
  EXPECT_FALSE(with(".safetensors"));             // suffix only, nothing before it
  // The valid shape still passes through the same path.
  EXPECT_TRUE(with("model-00001-of-00002.safetensors"));
}

TEST(SafetensorsIndex, IsSafeShardFilenameDirectly) {
  EXPECT_TRUE(is_safe_shard_filename("model-00001-of-00002.safetensors", 255));
  EXPECT_TRUE(is_safe_shard_filename("model.safetensors", 255));
  EXPECT_FALSE(is_safe_shard_filename("", 255));
  EXPECT_FALSE(is_safe_shard_filename("x.safetensors", 5));   // over the byte limit
  EXPECT_FALSE(is_safe_shard_filename("model.txt", 255));     // wrong suffix
  EXPECT_FALSE(is_safe_shard_filename("../m.safetensors", 255));
}

TEST(SafetensorsIndex, EnforcesLimits) {
  SafetensorsIndexLimits tiny;
  tiny.max_index_bytes = 16;
  EXPECT_FALSE(SafetensorsIndex::parse(kValid, tiny));  // index larger than the byte cap

  // Two distinct shards, but a limit of one.
  SafetensorsIndexLimits one_shard;
  one_shard.max_shards = 1;
  EXPECT_FALSE(SafetensorsIndex::parse(kValid, one_shard));

  SafetensorsIndexLimits one_entry;
  one_entry.max_entries = 1;
  EXPECT_FALSE(SafetensorsIndex::parse(kValid, one_entry));
}
