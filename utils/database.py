import sqlite3


def connect_db(db_path):
    """
    Utility function to get the database connection or create it if it doesn't exist.

    Parameters
    ----------
    db_path : string
        Path to the database's location on disk.

    Returns
    -------
    sqlite3 connection object
        Link to the database.
    """
    db_conn = sqlite3.connect(db_path)
    return db_conn


def create_db_tables(db_conn):
    """
    Ensures the necessary tables are present in the database when initialized.

    Parameters
    ----------
    db_conn : sqlite3 connection object
        Link to the database.
    """

    c = db_conn.cursor()

    c.execute(""" CREATE TABLE IF NOT EXISTS set_history(
        set_id text PRIMARY KEY,
        lifter text,
        lift text,
        weight real,
        nominal_diameter real,
        pixel_calibration real,
        number_of_reps real
        ); """)
    c.execute(""" CREATE TABLE IF NOT EXISTS rep_history(
        rep_id text PRIMARY KEY,
        set_id text REFERENCES set_history(set_id),
        lift text,
        average_velocity real,
        peak_velocity real,
        peak_power real,
        peak_height real,
        x_rom real,
        y_rom real,
        t_concentric real
        ); """)


def update_set_history(db_path, set_stats):
    """
    Commits the set statistics to the database.

    Parameters
    ----------
    set_stats : Dictionary
        Metadata for the set.
    """
    con = connect_db(db_path)
    c = con.cursor()
    # [:-1] is included here to avoid inserting the rep_stats dict, which is added last just before this is called
    sql = 'INSERT OR REPLACE INTO set_history(' \
          'set_id,lifter,lift,weight,nominal_diameter,pixel_calibration,number_of_reps) ' \
          'VALUES (?,?,?,?,?,?,?)'
    values = list(set_stats.values())[:-1]
    c.execute(sql, values)
    con.commit()
    con.close()


def update_rep_history(db_path, rep_stats):
    """
    Commits the rep statistics to the database.

    Parameters
    ----------
    rep_stats : Dictionary
        Metadata for each rep.
    """
    con = connect_db(db_path)
    c = con.cursor()
    for rep in rep_stats.keys():
        sql = 'INSERT OR REPLACE INTO rep_history(' \
              'rep_id,set_id,lift,average_velocity,peak_velocity,peak_power,peak_height,x_rom,y_rom,t_concentric) ' \
              'VALUES (?,?,?,?,?,?,?,?,?,?)'
        values = list(rep_stats[rep].values())
        c.execute(sql, values)
    con.commit()
    con.close()
