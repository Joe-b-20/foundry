"""Domain registry: name -> (pack, mold). Generic drivers (foreman,
islands, future sweeps) build from here and never import a domain
directly. Adding a domain = adding a branch here + a pack module that
honors the contract:

    pack.gate1(mold, tidy)          -> (score_tuple, cost_dict)
    pack.verify_trusted(mold, cand) -> (ok, details)

This refactor was earned, not speculative: judge v0 was sorting-shaped and
the PCF pack (second domain) demanded the polymorphism (RULES: no
abstraction until a second domain forces it).
"""


def build(domain, domain_params):
    if domain == "sorting_networks":
        from domains.sorting_networks import SortingNetworkPack
        from engine.molds import ComparatorMold
        pack = SortingNetworkPack(**domain_params)
        return pack, ComparatorMold(pack.n)
    if domain == "pcf":
        from domains.pcf import PCFPack
        from engine.molds_pcf import PCFMold
        return PCFPack(**domain_params), PCFMold()
    if domain == "bilinear":
        from domains.bilinear import BilinearPack
        from engine.molds_bilinear import BilinearMold
        return BilinearPack(**domain_params), BilinearMold()
    if domain == "bitmixer":
        from domains.bitmixer import BitMixerPack
        from engine.molds_bits import BitProgMold
        return BitMixerPack(**domain_params), BitProgMold()
    if domain == "rsqrt":
        from domains.rsqrt import RsqrtPack
        from engine.molds_float import FloatProgMold
        return RsqrtPack(**domain_params), FloatProgMold()
    if domain == "tanh":
        from domains.tanh import TanhPack
        from engine.molds_float import FloatProgMold
        n_const = domain_params.pop("n_const", 4)
        return TanhPack(**domain_params), FloatProgMold(n_const=n_const,
                                                        max_len=16)
    raise KeyError(f"unknown domain: {domain}")
