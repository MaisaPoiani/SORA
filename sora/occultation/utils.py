import numpy as np
import astropy.constants as const
import astropy.units as u

from sora.body import Body
from sora.ephem import EphemHorizons

__all__ = ['filter_negative_chord', 'positionv', 'add_arrow']


def positionv(star, ephem, observer, time):
    """Calculates the position and velocity of the occultation shadow relative
    to the observer.

    Parameters
    ----------
    star : `sora.Star`
        The coordinate of the star in the same reference frame as the ephemeris.
        It must be a Star object.

    ephem : `sora.Ephem`
        The object ephemeris. It must be an Ephemeris object.

    observer : `sora.Observer`
        The Observer information. It must be an Observer object.

    time : `astropy.time.Time`
        Reference instant to calculate position and velocity. It can be a string
        in the ISO format (yyyy-mm-dd hh:mm:ss.s) or an astropy Time object.

    Returns
    -------
    f, g, vf, vg : `list`
        The orthographic projection of the shadow relative to the observer.
        ``'f'`` is in the x-axis (East-West direction; East positive).
        ``'g'`` is in the y-axis (North-South direction; North positive).
    """
    from sora.ephem import EphemPlanete, EphemJPL, EphemKernel, EphemHorizons
    from sora.observer import Observer
    from sora.star import Star
    from astropy.time import Time

    if type(star) != Star:
        raise ValueError('star must be a Star object')
    if type(ephem) not in [EphemPlanete, EphemJPL, EphemKernel, EphemHorizons]:
        raise ValueError('ephem must be an Ephemeris object')
    if type(observer) != Observer:
        raise ValueError('observer must be an Observer object')
    time = Time(time)

    coord = star.geocentric(time)
    dt = 0.1*u.s

    if type(ephem) == EphemPlanete:
        ephem.fit_d2_ksi_eta(coord, verbose=False)
    ksio1, etao1 = observer.get_ksi_eta(time=time, star=coord)
    ksie1, etae1 = ephem.get_ksi_eta(time=time, star=coord)

    f = ksio1-ksie1
    g = etao1-etae1

    ksio2, etao2 = observer.get_ksi_eta(time=time+dt, star=coord)
    ksie2, etae2 = ephem.get_ksi_eta(time=time+dt, star=coord)

    nf = ksio2-ksie2
    ng = etao2-etae2

    vf = (nf-f)/0.1
    vg = (ng-g)/0.1

    return f, g, vf, vg


def filter_negative_chord(chord, chisquare, step=1, sigma=0):
    """Get points for the ellipse with the given input parameters.

    Parameters
    ----------
    chord : `sora.observer.Chord`
        Chord object, must be associated to an Occultation to work.

    chisquare : `sora.extra.ChiSquare`
        Resulted ChiSquare object of fit_ellipse.

    sigma : `int`, `float`
        Uncertainty of the expected ellipse, in km.

    step : `int`, `float`, `str`
        If a number, it corresponds to the step, in seconds, for each point of
        the chord path. The step can also be equal to ``'exposure'``. In this
        case, the chord path will consider the lightcurve individual times and
        exptime.
    """
    from sora.config.visuals import progressbar
    from sora.extra import ChiSquare
    from sora.extra.utils import get_ellipse_points

    keep = []
    if step == 'exposure':
        try:
            step = chord.lightcurve.exptime/10.
        except AttributeError:
            raise AttributeError('Chord.lightcurve does not have "exptime"')
        time_all = np.arange(chord.lightcurve.time.min(), chord.lightcurve.time.max(), step)
        time_exposure = np.array([])
        for i in range(len(chord.lightcurve.time)):
            event_model = (time_all > chord.lightcurve.time[i] - chord.lightcurve.exptime/2.) & (
                        time_all < chord.lightcurve.time[i] + chord.lightcurve.exptime/2.)
            time_exposure = np.append(time_exposure, time_all[event_model])
            f_all, g_all = chord.get_fg(time=time_exposure*u.s + chord.lightcurve.tref)
    else:
        f_all, g_all = chord.path(segment='full', step=step)
    for i in progressbar(range(len(chisquare.data['chi2'])), 'Filter chord: {}'.format(chord.name)):
        df_all = (f_all - chisquare.data['center_f'][i])
        dg_all = (g_all - chisquare.data['center_g'][i])

        r_all = np.sqrt(df_all**2 + dg_all**2)
        cut = r_all < 1.5*chisquare.data['equatorial_radius'][i]
        df_path = df_all[cut]
        dg_path = dg_all[cut]
        r_path = r_all[cut]
        theta_path = np.arctan2(dg_path, df_path)

        r_ellipse = get_ellipse_points(theta_path,
                                       equatorial_radius=chisquare.data['equatorial_radius'][i],
                                       oblateness=chisquare.data['oblateness'][i],
                                       center_f=chisquare.data['center_f'][i],
                                       center_g=chisquare.data['center_g'][i],
                                       position_angle=chisquare.data['position_angle'][i])[2]
        keep.append(np.all(r_path - r_ellipse + sigma > 0))

    filtered_chisquare = ChiSquare(chisquare.data['chi2'][keep], chisquare.npts,
                                   center_f=chisquare.data['center_f'][keep],
                                   center_g=chisquare.data['center_g'][keep],
                                   equatorial_radius=chisquare.data['equatorial_radius'][keep],
                                   oblateness=chisquare.data['oblateness'][keep],
                                   position_angle=chisquare.data['position_angle'][keep])
    return filtered_chisquare

def calc_geometric_albedo(equivalent_radius, H_obj, equivalent_radius_error=0, H_obj_error=0, H_sun=-26.74, verbose=True):
    """ Calculate the geometric albedo.

        Parameters
        ----------
        equivalent_radius : `float`, `int`
            Equivalent radius of the occulting body, in km.

        H_obj : `float`, `int`
            Occulting body's absolute magnitude.
        
        equivalent_radius_error : `float`, `int`, default=0
            Equivalent radius uncertainty of the occulting body, in km.

        H_obj_error : `float`, `int`, default=0
            Occulting body's absolute magnitude uncertainty.

        H_sun : `float`, `int`, default=-26.74
            Sun absolute magnitude.

        verbose : `bool`, default is True
            If True, it prints text.
     Returns
    -------
        geometric_albedo : `float`
            Geometric albedo

        delta_albedo : `float`
            Geometric albedo uncertainty
    """
    geometric_albedo = (10**(0.4*(H_sun - H_obj))) * ((u.au.to('km')/equivalent_radius)**2)
    H_obj_error = np.absolute(H_obj_error)
    equivalent_radius_error = np.absolute(equivalent_radius_error)
    albedo_error_p = (10**(0.4*(H_sun - H_obj+H_obj_error))) * ((u.au.to('km')/(equivalent_radius+equivalent_radius_error))**2)    
    albedo_error_m = (10**(0.4*(H_sun - H_obj-H_obj_error))) * ((u.au.to('km')/(equivalent_radius-equivalent_radius_error))**2)
    delta_albedo = np.absolute(albedo_error_p - albedo_error_m)
    if verbose:
        if (H_obj_error != 0) or (equivalent_radius_error != 0):
            print('geometric albedo: {:.3f} +/- {:.3f} \n'.format(geometric_albedo, delta_albedo))
        else:
            print('geometric albedo: {:.3f} \n'.format(geometric_albedo))
    return geometric_albedo, delta_albedo


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """ Add an arrow to a chord.

        Parameters
        ----------
        line : `Line2D object`
            Line2D object

        position : `float`, `int`
            x-position of the arrow. If None, mean of xdata is taken.

        direction : `string`, default='right'
            'left' or 'right'

        size : `float`, `int`, default=15
            Size of the arrow in fontsize points.

        color : `string`, default=None
            If None, line color is taken.

        See Also
        --------
        https://stackoverflow.com/a/34018322/3137585
    """

    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )


def calc_sun_dif_ld(body_coord, star_coord, time, observer="geocenter"):
    """Computes differential light deflection of star and body caused by the Sun.

    Parameters
    ----------
    body_coord : `astropy.coordinates.SkyCoord`
        The astrometric coordinate of the body at given time

    star_coord : `astropy.coordinates.SkyCoord`
        The astrometric coordinate of the star at given time

    time : `astropy.time.Time`
        The time which to compute the light deflection.
        Use to compute the position of the Sun

    observer : `sora.observer.Observer`
        The observer to compute the position of the Sun.

    Returns
    -------
    ld_ra, ld_dec : `astropy.coordinates.Angle`
        The offsets in right ascension and declination of the object relative to the star,
        considering the light deflection caused by the Sun.

    """
    import erfa
    from astropy.coordinates import SkyCoord

    sun = Body(name='Sun', spkid='10', ephem=EphemHorizons(name='Sun', spkid=10, id_type='majorbody'),
               GM=const.G * (1 * u.M_sun), database=None)
    pos_sun = sun.get_position(time=time, observer=observer)
    bm = 1  # Sun Mass in Solar Masses

    e = -(pos_sun.cartesian.xyz / pos_sun.cartesian.norm()).value  # unitary vector Sun -> observer
    em = pos_sun.distance.to(u.AU).value  # distance Sun -> observer in AU

    pbody = (body_coord.cartesian.xyz / body_coord.cartesian.norm()).value  # unitary vector observer -> body
    pbs = (body_coord.cartesian - pos_sun.cartesian)  # vector Sun -> body
    q = (pbs.xyz / pbs.norm()).value  # unitary vector Sun -> body

    # computes the gravitational light deflection, return new normalized vector observer -> body
    ds_body = erfa.ld(bm, pbody, q, e, em, 0)

    os_body = SkyCoord(*ds_body * body_coord.cartesian.norm(), representation_type='cartesian')
    dra_b, ddec_b = body_coord.spherical_offsets_to(os_body)  # computes offset between corrected and astrometric

    pstar = (star_coord.cartesian.xyz / star_coord.cartesian.norm()).value  # unitary vector observer -> star
    pbs = (star_coord.cartesian - pos_sun.cartesian)  # vector Sun -> star
    q = (pbs.xyz / pbs.norm()).value  # unitary vector Sun -> star

    # computes the gravitational light deflection, return new normalized vector observer -> body
    ds_star = erfa.ld(bm, pstar, q, e, em, 0)

    os_star = SkyCoord(*ds_star * star_coord.cartesian.norm(), representation_type='cartesian')
    dra_s, ddec_s = star_coord.spherical_offsets_to(os_star)  # computes offset between corrected and astrometric

    return dra_b - dra_s, ddec_b - ddec_s

def calc_massives_ld(body_coord, star_coord, time, observer, massives=['sun', 'jupiter', 'saturn']):
    #the standard massive list contains Sun, Jupiter and Saturn
    
    #new imports
    from sora.observer import Observer
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, CartesianRepresentation
    import pandas as pd
    from sora.ephem import EphemHorizons

    G = const.G.to(u.AU**3 / u.kg / u.day**2)
    c = const.c.to(u.AU/u.day)
    gamma = 1

    #if it's not informed a time format, assume UTC
    if not isinstance(time, Time):
        time = Time(time, scale='utc')
        
    #defining massive bodys of Solar System
    massive_data = {
        'sun': {'name': 'Sun', 'spkid': 10, 'mass': 1 * u.M_sun},
        'mercury': {'name': 'Mercury', 'spkid': 199, 'mass': 3.3011e23 * u.kg},
        'venus': {'name': 'Venus', 'spkid': 299, 'mass': 4.8675e24 * u.kg},
        'earth': {'name': 'Earth', 'spkid': 399, 'mass': const.M_earth},
        'mars': {'name': 'Mars', 'spkid': 499, 'mass': 6.4171e23 * u.kg},
        'jupiter': {'name': 'Jupiter', 'spkid': 599, 'mass': const.M_jup},
        'saturn': {'name': 'Saturn', 'spkid': 699, 'mass': 5.6834e26 * u.kg},
        'uranus': {'name': 'Uranus', 'spkid': 799, 'mass': 8.6810e25 * u.kg},
        'neptune': {'name': 'Neptune', 'spkid': 899, 'mass': 1.024e26 * u.kg}
    }

    #creating a list of massives from their names
    massives_list = []
    
    for massive_name in massives:
        massive_name_lower = massive_name.lower().strip()

        if massive_name_lower in massive_data:
            data = massive_data[massive_name_lower]
            body = Body(
                name=data['name'],
                spkid=str(data['spkid']),
                ephem=EphemHorizons(
                    name=data['spkid'], #to ensure unique identification
                    spkid=data['spkid'],
                    id_type='majorbody'
                ),
                GM=const.G * data['mass'],
                database=None
            )
            massives_list.append(body)

        else:
            print(f"Sorry, '{massive_name}' it's not a valid massive body. Available options: {', '.join(massive_data.keys())}")

    #eq 70 Klioner (2003)
    def explicit_delta_k_pN(G, gamma, M_a, c, R, r_eA, r_oA, r_eA_uni, r_oA_uni):
        term1 = (((1 + gamma) * G * M_a) / c**2)
        numerator = R.cross(r_eA.cross(r_oA))
        denominator = R.norm() * r_oA.norm() * (r_eA.norm() * r_oA.norm() + r_oA.dot(r_eA))
        return -term1 * (numerator / denominator)

    def deflection(observador, fonte, massive, mass):
        r_oA = observador - massive
        r_oA_uni = r_oA / r_oA.norm()
        r_eA = fonte - massive
        r_eA_uni = r_eA / r_eA.norm()
        R = observador - fonte
        R_uni = CartesianRepresentation(R.x / R.norm(), R.y / R.norm(), R.z / R.norm())
        vecDeflection = explicit_delta_k_pN(G, gamma, mass, c, R, r_eA, r_oA, r_eA_uni, r_oA_uni)
        return vecDeflection, R_uni


    table_lines = [] #empty table

    #getting bodys positions
    cart_body = body_coord.cartesian
    cart_body_AU = CartesianRepresentation(x=cart_body.x.to(u.AU),
                                                y=cart_body.y.to(u.AU), z=cart_body.z.to(u.AU))

    cart_star =  star_coord.cartesian
    cart_star_AU = CartesianRepresentation(x=cart_star.x.to(u.AU),
                                               y=cart_star.y.to(u.AU), z=cart_star.z.to(u.AU))

    r_obs = observer.get_vector(time)
    cart_observer = r_obs.cartesian
    cart_observer_AU = CartesianRepresentation(x=cart_observer.x.to(u.AU),
                                                 y=cart_observer.y.to(u.AU), z=cart_observer.z.to(u.AU))

    #the initial deflection is null, vectorially adding for each massive body
    vecDeflectionBody_total = CartesianRepresentation(0, 0, 0)
    vecDeflectionStar_total = CartesianRepresentation(0, 0, 0)

    for massive in massives_list:
        pos_ephem_massive = massive.get_position(time=time, observer=observer)
        cart_massive = pos_ephem_massive.cartesian
        cart_massive_AU = CartesianRepresentation(x=cart_massive.x.to(u.AU), y=cart_massive.y.to(u.AU), z=cart_massive.z.to(u.AU))
        M_massive = massive.GM/G

        vecDeflectionStar, R_uni = deflection(cart_observer_AU, cart_body_AU, cart_massive_AU, M_massive)
        vecDeflectionBody, _ = deflection(cart_observer_AU, cart_star_AU, cart_massive_AU, M_massive)
        vecDeflectionBody_total += vecDeflectionStar
        vecDeflectionStar_total += vecDeflectionBody

    #the relative light deflection that affects the occulting body
    vecDeflectionSub = vecDeflectionStar_total - vecDeflectionBody_total
    sinthetaSub = (vecDeflectionSub).norm().decompose()
    deflection_sub = np.arcsin(sinthetaSub)
    astrometric_correction = cart_body_AU.norm() * np.sin(deflection_sub)

    #getting the apparent position of related bodys
    v = cart_body_AU - cart_observer_AU
    v_unit = v / v.norm()
    v_apparent = v_unit + vecDeflectionSub
    v_apparent = v_apparent / v_apparent.norm()
    pos_apparent_body = cart_observer_AU + v_apparent * v.norm()
    coord_apparent_body = SkyCoord(pos_apparent_body, frame='icrs', representation_type='cartesian')
    coord_body = SkyCoord(cart_body_AU, frame='icrs', representation_type='cartesian')
    dra_apparent_body, ddec_apparent_body = coord_apparent_body.spherical_offsets_to(coord_body)

    v_s = cart_star_AU - cart_observer_AU
    v_unit_s = v_s / v_s.norm()
    v_apparent_s = v_unit_s + vecDeflectionSub
    v_apparent_s = v_apparent_s / v_apparent_s.norm()
    pos_apparent_star = cart_observer_AU + v_apparent_s * v_s.norm()
    coord_apparent_star = SkyCoord(pos_apparent_star, frame='icrs', representation_type='cartesian')
    coord_star = SkyCoord(cart_star_AU, frame='icrs', representation_type='cartesian')
    dra_apparent_star, ddec_apparent_star = coord_apparent_star.spherical_offsets_to(coord_star)

    #building the returning table
    table_lines.append({
        'body': 'Occulting',
        'RA ephem (deg)': f"{coord_body.spherical.lon.deg:.10f}",
        'DEC ephem (deg)': f"{coord_body.spherical.lat.deg:.10f}",
        'astrometric correction (km)': f"{astrometric_correction.to(u.km).value:.10f}",
        'RA apparent (deg)': f"{coord_apparent_body.spherical.lon.deg:.10f}",
        'DEC apparent (deg)': f"{coord_apparent_body.spherical.lat.deg:.10f}",
        'dRA (mas)': f"{dra_apparent_body.mas:.10f}",
        'dDEC (mas)': f"{ddec_apparent_body.mas:.10f}"
    })

    table_lines.append({
        'body': 'Star',
        'RA ephem (deg)': f"{coord_star.spherical.lon.deg:.10f}",
        'DEC ephem (deg)': f"{coord_star.spherical.lat.deg:.10f}",
        'astrometric correction (km)': f"{astrometric_correction.to(u.km).value:.10f}",
        'RA apparent (deg)': f"{coord_apparent_star.spherical.lon.deg:.10f}",
        'DEC apparent (deg)': f"{coord_apparent_star.spherical.lat.deg:.10f}",
        'dRA (mas)': f"{dra_apparent_star.mas:.10f}",
        'dDEC (mas)': f"{ddec_apparent_star.mas:.10f}"
    })

    #showing each massive body information
    for massive in massives_list:
        pos_ephem_massive = massive.get_position(time=time, observer=observer)
        cart_massive = pos_ephem_massive.cartesian
        cart_massive_AU = CartesianRepresentation(x=cart_massive.x.to(u.AU), y=cart_massive.y.to(u.AU), z=cart_massive.z.to(u.AU))
        coord_massive = SkyCoord(cart_massive_AU, frame='icrs', representation_type='cartesian')
        table_lines.append({
            'body': f"massive {massive.name}",
            'RA ephem (deg)': f"{coord_massive.spherical.lon.deg:.10f}",
            'DEC ephem (deg)': f"{coord_massive.spherical.lat.deg:.10f}",
            'astrometric correction (km)': '',
            'RA apparent (deg)': '',
            'DEC apparent (deg)': '',
            'dRA (mas)': '',
            'dDEC (mas)': ''
        })

    #returning the complete table
    df = pd.DataFrame(table_lines)
    return df
