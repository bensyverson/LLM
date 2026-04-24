//
//  LLM+Tools.swift
//  LLM
//
//  Created by Claude on 2026-01-30.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    /// The type of tool. Currently only function calling is supported.
    enum ToolType: String, Friendly {
        /// A function that the model can call.
        case function
    }

    /// Defines a tool the model can call, wrapping a ``FunctionDefinition``.
    struct ToolDefinition: Friendly {
        public var type: ToolType
        public var function: FunctionDefinition

        public init(
            type: ToolType = .function,
            function: FunctionDefinition,
        ) {
            self.type = type
            self.function = function
        }
    }

    /// Describes a function the model can call, including its name, description, and parameter schema.
    struct FunctionDefinition: Friendly {
        /// The function name the model will use to identify this tool.
        public var name: String
        /// A description of what the function does, helping the model decide when to call it.
        public var description: String
        /// JSON Schema describing the function's parameters.
        public var parameters: JSONSchema

        public init(
            name: String,
            description: String,
            parameters: JSONSchema,
        ) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    /// A JSON Schema definition used to describe tool parameters.
    ///
    /// Use the static convenience methods (``object(properties:required:description:)``,
    /// ``string(description:enum:)``, ``integer(description:)``, etc.) for ergonomic construction.
    final class JSONSchema: Codable, Hashable, Equatable, Sendable {
        public enum SchemaType: String, Friendly {
            case object, array, string, number, integer, boolean, null
        }

        public let type: SchemaType
        public let properties: [String: JSONSchema]?
        public let items: JSONSchema?
        public let required: [String]?
        public let description: String?
        public let `enum`: [String]?

        public init(
            type: SchemaType,
            properties: [String: JSONSchema]? = nil,
            items: JSONSchema? = nil,
            required: [String]? = nil,
            description: String? = nil,
            enum: [String]? = nil,
        ) {
            self.type = type
            self.properties = properties
            self.items = items
            self.required = required
            self.description = description
            self.enum = `enum`
        }

        public static func == (lhs: JSONSchema, rhs: JSONSchema) -> Bool {
            lhs.type == rhs.type &&
                lhs.properties == rhs.properties &&
                lhs.items == rhs.items &&
                lhs.required == rhs.required &&
                lhs.description == rhs.description &&
                lhs.enum == rhs.enum
        }

        public func hash(into hasher: inout Hasher) {
            hasher.combine(type)
            hasher.combine(properties)
            hasher.combine(items)
            hasher.combine(required)
            hasher.combine(description)
            hasher.combine(`enum`)
        }

        /// Convenience initializers for common types
        public static func object(
            properties: [String: JSONSchema],
            required: [String]? = nil,
            description: String? = nil,
        ) -> JSONSchema {
            JSONSchema(type: .object, properties: properties, required: required, description: description)
        }

        public static func array(items: JSONSchema, description: String? = nil) -> JSONSchema {
            JSONSchema(type: .array, items: items, description: description)
        }

        public static func string(description: String? = nil, enum enumValues: [String]? = nil) -> JSONSchema {
            JSONSchema(type: .string, description: description, enum: enumValues)
        }

        public static func number(description: String? = nil) -> JSONSchema {
            JSONSchema(type: .number, description: description)
        }

        public static func integer(description: String? = nil) -> JSONSchema {
            JSONSchema(type: .integer, description: description)
        }

        public static func boolean(description: String? = nil) -> JSONSchema {
            JSONSchema(type: .boolean, description: description)
        }
    }

    /// Controls how the model selects tools.
    enum ToolChoice: Friendly {
        /// Let the model decide whether to call a tool.
        case auto
        /// The model must call at least one tool.
        case required
        /// The model must not call any tools.
        case none
        /// The model must call the named tool.
        case tool(name: String)

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let stringValue = try? container.decode(String.self) {
                switch stringValue {
                case "auto": self = .auto
                case "required": self = .required
                case "none": self = .none
                default: throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unknown tool choice: \(stringValue)")
                }
            } else {
                let objectContainer = try decoder.container(keyedBy: ToolChoiceCodingKeys.self)
                let functionContainer = try objectContainer.nestedContainer(keyedBy: FunctionCodingKeys.self, forKey: .function)
                let name = try functionContainer.decode(String.self, forKey: .name)
                self = .tool(name: name)
            }
        }

        public func encode(to encoder: Encoder) throws {
            switch self {
            case .auto:
                var container = encoder.singleValueContainer()
                try container.encode("auto")
            case .required:
                var container = encoder.singleValueContainer()
                try container.encode("required")
            case .none:
                var container = encoder.singleValueContainer()
                try container.encode("none")
            case let .tool(name):
                var container = encoder.container(keyedBy: ToolChoiceCodingKeys.self)
                try container.encode("function", forKey: .type)
                var functionContainer = container.nestedContainer(keyedBy: FunctionCodingKeys.self, forKey: .function)
                try functionContainer.encode(name, forKey: .name)
            }
        }

        private enum ToolChoiceCodingKeys: String, CodingKey {
            case type, function
        }

        private enum FunctionCodingKeys: String, CodingKey {
            case name
        }
    }

    /// A tool call requested by the model, containing the function name and arguments.
    struct ToolCall: Friendly {
        /// The unique identifier for this tool call (used to match results back).
        public var id: String
        /// The type of tool call (always `"function"`).
        public var type: String
        /// The function name and JSON-encoded arguments.
        public var function: FunctionCall

        public init(
            id: String,
            type: String = "function",
            function: FunctionCall,
        ) {
            self.id = id
            self.type = type
            self.function = function
        }
    }

    /// The name and arguments of a function call requested by the model.
    struct FunctionCall: Friendly {
        /// The name of the function to call.
        public var name: String
        /// The function arguments as a JSON string.
        public var arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }
}
