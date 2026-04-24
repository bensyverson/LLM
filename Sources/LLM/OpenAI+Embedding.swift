//
//  OpenAI+Embedding.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    /// The request body for an embedding API call.
    struct EmbeddingRequest: Codable {
        public enum EncodingFormat: String, Codable {
            case float, base64
        }

        public enum EmbeddingModel: String, Codable {
            case textEmbedding3Large = "text-embedding-3-large"
            case textEmbedding3Small = "text-embedding-3-small"
            case textEmbedding2SmallOlder = "text-embedding-ada-002"
        }

        public var input: String
        public var model: EmbeddingModel
        public var encoding_format: EncodingFormat
        public var dimensions: Int?

        public init(
            input: String,
            model: EmbeddingModel,
            encoding_format: EncodingFormat,
            dimensions: Int? = nil,
        ) {
            self.input = input
            self.model = model
            self.encoding_format = encoding_format
            self.dimensions = dimensions
        }
    }

    /// The response from an embedding API call.
    struct EmbeddingList: Codable {
        public struct EmbeddingData: Codable {
            public var embedding: [Float]
        }

        public var data: [EmbeddingData]

        public init(data: [EmbeddingData]) {
            self.data = data
        }
    }
}
