from app.services.places import (
    INLAND_CITIES,
    PLACES,
    infer_map_position,
    in_known_water,
    inland_placeholder,
    match_place,
    offset_on_land,
)


def test_old_rectangular_grid_put_cameras_in_the_gulfs():
    """The previous 6x6 scatter covered Gulf of Khambhat and Gulf of Kutch."""

    def old_grid(n: int) -> tuple[float, float]:
        col = n % 6
        row = (n // 6) % 6
        lat = 21.5 + row * 0.45 + ((n * 17) % 10) * 0.012
        lng = 69.7 + col * 0.55 + ((n * 13) % 10) * 0.012
        return lat, lng

    assert in_known_water(*old_grid(5))  # cam05 Visat was in the Gulf of Khambhat
    assert in_known_water(*old_grid(11))  # cam11 Dolatpara was in the Gulf of Khambhat
    assert in_known_water(*old_grid(12))  # cam12 Adalaj was in the Gulf of Kutch
    assert in_known_water(*old_grid(18))  # cam18 Rajkot was in the Gulf of Kutch


def test_live_catalogue_names_resolve_to_land():
    samples = {
        "cam05": "05 Visat teen Rasta",
        "cam06": "06 Timbavadi gate-Junagadh",
        "cam07": "07 hero-showroom-gir-somnath",
        "cam12": "12 Tri Mandir Adalaj Tollnaka",
        "cam17": "17 Rajkot Bus Port CCTV",
        "cam18": "18 Rajkot CCTV",
        "cam19": "19 KHAPARIA GRAM PANCHAYAT , TALUKA GANDEVI, DISTRICT NAVSARI",
        "cam21": "23 Patan Dethali Char Rasta",
        "cam24": "33 dehgam",
        "cam27": "36 bilimora",
        "cam30": "Gandhidham Rambaugh p2",
    }
    expected = {
        "cam05": "Ahmedabad",
        "cam06": "Junagadh",
        "cam07": "Gir Somnath",
        "cam12": "Adalaj",
        "cam17": "Rajkot",
        "cam18": "Rajkot",
        "cam19": "Gandevi",
        "cam21": "Patan",
        "cam24": "Dehgam",
        "cam27": "Bilimora",
        "cam30": "Gandhidham",
    }
    for cam_id, name in samples.items():
        pos = infer_map_position(cam_id, name=name)
        assert pos.source == "inferred_place"
        assert pos.label == expected[cam_id]
        assert not in_known_water(pos.lat, pos.lng)


def test_unnamed_placeholder_is_inland_and_stable():
    a = inland_placeholder("cam99")
    b = inland_placeholder("cam99")
    c = inland_placeholder("cam98")
    assert a.source == "placeholder"
    assert a == b
    assert (a.lat, a.lng) != (c.lat, c.lng)
    assert not in_known_water(a.lat, a.lng)
    assert not in_known_water(c.lat, c.lng)


def test_gazetteer_centroids_and_offsets_stay_on_land():
    for place in PLACES:
        assert not in_known_water(place.lat, place.lng), place.label
        for i in range(12):
            lat, lng = offset_on_land(f"cam{i:02d}", place.lat, place.lng)
            assert not in_known_water(lat, lng), (place.label, i)
    for label, lat, lng in INLAND_CITIES:
        assert not in_known_water(lat, lng), label


def test_match_place_prefers_longer_alias():
    assert match_place("07 hero-showroom-gir-somnath").label == "Gir Somnath"
    assert match_place("04 Paldi Circle").label == "Ahmedabad"
    assert match_place("unknown junction") is None
