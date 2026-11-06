#pragma once

#include "core/strict_value.h"
#include "ninfer/engine_options.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline constexpr const char* kHybridExpertHelp =
    "  Host and disk expert options (--expert-residency host|disk):\n"
    "  --expert-dma-share F|auto     GPU share of a decode or verification call's cache\n"
    "                                misses, 0..1; the CPU computes the rest (GGUF and\n"
    "                                Qwen3.5/3.6 MoE experts: the GPU's results bit for bit).\n"
    "                                auto (default): those split each call by measured CPU\n"
    "                                and bus times, Flash-Next native experts and DeepSeek-V4.1\n"
    "                                disk experts copy every miss. Flash-Next, Gemma 4, Flint\n"
    "                                and DeepSeek-V4.1 calls of more than 8 tokens, and\n"
    "                                Qwen3.5/3.6 calls routed to half the experts or more,\n"
    "                                always use the GPU\n"
    "  --expert-cpu-threads N        CPU workers, 1..256 (default: one per physical core, at\n"
    "                                most half the CPUs)\n"
    "  --expert-profile FILE         fill the expert cache from recorded counts (default: the\n"
    "                                built-in profile of the model family, where one exists)\n"
    "  --expert-profile-out FILE     record expert counts for this artifact after requests\n"
    "  --expert-host-mib N           DeepSeek-V4.1 disk experts: keep up to N MiB of the\n"
    "                                experts the routing profile ranks highest in pinned host\n"
    "                                memory (copied like host experts) and read the rest from\n"
    "                                the files with direct I/O (default 0: all from the files)\n"
    "  Flash-Next-only expert options:\n"
    "  --expert-misses staged|mapped GGUF experts: copy misses into device staging while the\n"
    "                                cached experts run (default), or read host experts across\n"
    "                                the bus inside the expert kernels\n"
    "  --expert-tail-skip N          GGUF host experts: leave out of a decode or verification\n"
    "                                call each uncached expert that all its tokens rank past\n"
    "                                place N (1..9) of their top ten; changes results\n"
    "  --expert-prefill-buffers 1|2  GGUF host experts: device buffers for a long prompt's\n"
    "                                uncached experts; 2 copies the next layer's while this one\n"
    "                                runs (default 2 with the automatic cache, which holds them\n"
    "                                as expert cache slots between prompts, else 1)\n"
    "  --expert-cache-adaptive       replace cold cached experts between calls;\n"
    "                                changes the CPU/GPU arithmetic partition\n"
    "  DeepSeek-V4.1-only delivery experiments (T0833; each off by default, each also set by\n"
    "  the NINFER_DS_* variable named beside it, 1|on|true or 0|off|false):\n"
    "  --deepseek-prefetch-ahead on|off     a wide prompt call queues the next block's expert\n"
    "                                copies behind its own and names the block from the pass's\n"
    "                                real order (NINFER_DS_PREFETCH_AHEAD)\n"
    "  --deepseek-cache-settle on|off       after a prompt, its routes feed the expert tier and\n"
    "                                swap rounds run at once; again while the Engine idles\n"
    "                                (NINFER_DS_CACHE_SETTLE)\n"
    "  --deepseek-cache-settle-ms N         the most a settle after a prompt takes (default 400;\n"
    "                                NINFER_DS_CACHE_SETTLE_MS)\n"
    "  --deepseek-cache-settle-idle-ms N    the most an idle settle takes (default 6000;\n"
    "                                NINFER_DS_CACHE_SETTLE_IDLE_MS)\n"
    "  --deepseek-short-prompt on|off       a prompt of at most N new tokens runs in rows of 8\n"
    "                                through the narrow expert path (NINFER_DS_SHORT_PROMPT)\n"
    "  --deepseek-short-prompt-tokens N     that N (default 96; NINFER_DS_SHORT_PROMPT_TOKENS)\n"
    "  --deepseek-profile-weights on|off    --expert-profile counts also seed the expert tier's\n"
    "                                counts (NINFER_DS_PROFILE_WEIGHTS)\n";

template<class Value>
bool parse_hybrid_expert_option(std::string_view option, HybridExpertOptions& out, Value&& value) {
    if (option == "--expert-dma-share") {
        const std::string text(value());
        if (text == "auto") {
            out.dma_share.reset();
            return true;
        }
        float share = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), share);
        if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(share) || share < 0 || share > 1) {
            throw std::invalid_argument("--expert-dma-share takes a finite value in 0..1 or auto");
        }
        out.dma_share = share;
    } else if (option == "--expert-cpu-threads") {
        const std::string text(value());
        std::uint32_t threads = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), threads);
        if (error != std::errc{} || end != text.data() + text.size() || threads == 0 || threads > 256) {
            throw std::invalid_argument("--expert-cpu-threads takes an integer in 1..256");
        }
        out.cpu_threads = threads;
    } else if (option == "--expert-tail-skip") {
        const std::string text(value());
        std::uint32_t rank      = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), rank);
        if (error != std::errc{} || end != text.data() + text.size() || rank == 0 || rank > 9) {
            throw std::invalid_argument("--expert-tail-skip takes an integer in 1..9");
        }
        out.tail_skip_rank = rank;
    } else if (option == "--expert-prefill-buffers") {
        const std::string_view count(value());
        if (count != "1" && count != "2") {
            throw std::invalid_argument("--expert-prefill-buffers takes 1 or 2");
        }
        out.prefill_buffers = count == "2" ? 2 : 1;
    } else if (option == "--expert-misses") {
        const std::string_view mode(value());
        if (mode == "staged") {
            out.mapped_misses = false;
        } else if (mode == "mapped") {
            out.mapped_misses = true;
        } else {
            throw std::invalid_argument("--expert-misses takes staged or mapped");
        }
    } else if (option == "--expert-cache-adaptive") {
        out.adaptive_cache = true;
    } else if (option == "--deepseek-prefetch-ahead" || option == "--deepseek-cache-settle" ||
               option == "--deepseek-short-prompt" || option == "--deepseek-profile-weights") {
        // On the command line only on|off; the environment also takes 1|0|true|false.
        const std::string text(value());
        if (text != "on" && text != "off") {
            throw std::invalid_argument(std::string(option) + " takes on or off");
        }
        const bool on = text == "on";
        if (option == "--deepseek-prefetch-ahead") { out.deepseek_prefetch_ahead = on; }
        else if (option == "--deepseek-cache-settle") { out.deepseek_cache_settle = on; }
        else if (option == "--deepseek-short-prompt") { out.deepseek_short_prompt = on; }
        else { out.deepseek_profile_weights = on; }
    } else if (option == "--deepseek-cache-settle-ms" || option == "--deepseek-cache-settle-idle-ms" ||
               option == "--deepseek-short-prompt-tokens") {
        const std::string text(value());
        const auto count  = parse_count(text);
        const bool tokens = option == "--deepseek-short-prompt-tokens";
        if (!count || (tokens ? (*count == 0 || *count > 4096) : *count > 600000)) {
            throw std::invalid_argument(std::string(option) + (tokens
                ? " takes an integer in 1..4096" : " takes milliseconds in 0..600000"));
        }
        if (tokens) { out.deepseek_short_prompt_tokens = *count; }
        else if (option == "--deepseek-cache-settle-ms") { out.deepseek_cache_settle_ms = *count; }
        else { out.deepseek_cache_settle_idle_ms = *count; }
    } else if (option == "--expert-host-mib") {
        const auto mib = parse_count(std::string(value()));
        if (!mib || *mib > (1u << 24)) {
            throw std::invalid_argument("--expert-host-mib takes MiB in 0..16777216");
        }
        out.expert_host_bytes = std::uint64_t(*mib) << 20;
    } else if (option == "--expert-profile" || option == "--expert-profile-out") {
        const std::string path(value());
        if (path.empty()) { throw std::invalid_argument(std::string(option) + " takes a file path"); }
        (option == "--expert-profile" ? out.routing_profile : out.record_profile) = path;
    } else { return false; }
    return true;
}

} // namespace ninfer::product
